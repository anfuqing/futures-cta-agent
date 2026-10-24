"""
郑商所杯 · 实盘信号工具（signal_tool.py）
========================================
作用：读取最新期货日线 + 你当前的持仓，输出"下一交易日的傻瓜式操作指令"。
     指令只给计划；实际成交一律按次日开盘价执行。

【每次运行前维护 positions.json】——记录你现在手上有什么仓：
{
  "CF0": {"side": -1, "entry": 15830.0, "units": 1, "stop": 16268.0},
  "OI0": {"side":  1, "entry": 10300.0, "units": 2, "stop":  9990.0}
}
  side:  1 = 多单，-1 = 空单
  entry: 开仓价
  units: 手数
  stop:  止损价（开仓当天工具会告诉你，抄过来填上）
没有任何持仓就写 {}（空）。

【输出三块】
 1) 明日开仓  ：收盘突破前55日通道的品种 → 次日开盘开仓，方向/手数/参考止损
 2) 持仓处理  ：继续持有 / 次日平仓（20日通道退出）/ 盘中止损价
 3) 空仓观望  ：未触发信号的品种，明日不操作

【注意】
 - 手数 = 账户×0.8% ÷ (2×ATR×乘数)，不足1手不交易；保证金占用≤权益80%
 - 止损价开仓后按「实际开盘价 ± 2×ATR」最终确定，盘中触及即平
 - 本工具按定稿参数（0.8%风险、最多5仓、双向、无均线/ADX过滤）输出
========================================
"""
import json, os, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import akshare as ak

# ================= 配置（定稿） =================
EQUITY = 500000
RISK_PCT = 0.008
FEE_MULT = 1.5
FETCH_START = "20230701"
ENTRY_N, EXIT_N, ATR_N, STOP_K = 55, 20, 20, 2.0
MAX_POS = 5
LONG_ONLY = False
ACTIVE = []

FUTURES = {
    "SA0": ("纯碱", 20, 0.15, 3.0), "FG0": ("玻璃", 20, 0.15, 6.0),
    "MA0": ("甲醇", 10, 0.12, 2.0), "TA0": ("PTA", 5, 0.10, 3.0),
    "RM0": ("菜粕", 10, 0.10, 1.5), "SR0": ("白糖", 10, 0.09, 3.0),
    "CF0": ("棉花", 5, 0.10, 4.3), "AP0": ("苹果", 10, 0.10, 5.0),
    "UR0": ("尿素", 20, 0.10, 5.0), "OI0": ("菜油", 10, 0.10, 2.0),
    "PF0": ("短纤", 5, 0.10, 3.0), "CJ0": ("红枣", 5, 0.10, 3.0),
    "PK0": ("花生", 5, 0.10, 4.0), "SF0": ("硅铁", 5, 0.12, 3.0),
    "SM0": ("锰硅", 5, 0.12, 3.0), "CY0": ("棉纱", 5, 0.10, 4.0),
    "SH0": ("烧碱", 30, 0.12, 2.0), "PX0": ("对二甲苯", 5, 0.10, 3.0),
}


def back_adjust(df, thresh=0.08):
    p = df[["open", "high", "low", "close"]].astype(float).to_numpy().copy()
    for i in range(1, len(p)):
        if p[i - 1, 3] > 0 and abs(p[i, 3] / p[i - 1, 3] - 1) > thresh:
            p[i:] *= p[i - 1, 3] / p[i, 3]
    df[["open", "high", "low", "close"]] = p
    return df


def fetch_futures(symbol):
    """获取新浪主力连续日线；清洗数据并避免收盘前误用当天未完成日线。"""
    wait = 10
    last_error = None
    for attempt in range(1, 5):
        try:
            raw = ak.futures_main_sina(symbol=symbol)
            if raw is None or raw.empty:
                raise RuntimeError(f"{symbol} 接口返回空数据")

            # 逐个寻找字段，避免某个字段名缺失时 next() 报难懂的 StopIteration。
            aliases = {
                "日期": ["日期", "date", "datetime"],
                "open": ["开盘价", "open"],
                "high": ["最高价", "high"],
                "low": ["最低价", "low"],
                "close": ["收盘价", "close"],
            }
            rename = {}
            for target, candidates in aliases.items():
                found = next((c for c in candidates if c in raw.columns), None)
                if found is None:
                    raise RuntimeError(f"{symbol} 接口缺少字段 {target}，实际字段：{list(raw.columns)}")
                rename[found] = target
            df = raw.rename(columns=rename)[["日期", "open", "high", "low", "close"]].copy()
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df = df.dropna(subset=["日期", "open", "high", "low", "close"])
            df = df[df["日期"] >= pd.to_datetime(FETCH_START)]
            df = df.sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
            # 排除明显不可能的 OHLC 行，防止异常值污染 ATR 和通道。
            valid = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
            valid &= df["high"] >= df[["open", "low", "close"]].max(axis=1)
            valid &= df["low"] <= df[["open", "high", "close"]].min(axis=1)
            df = df.loc[valid].reset_index(drop=True)
            if df.empty:
                raise RuntimeError(f"{symbol} 清洗后无有效数据")

            # 以中国时区为准。15:30 前运行时，排除当天可能尚未完成的日线。
            now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
            today = pd.Timestamp(now_cn.date())
            if now_cn.hour < 15 or (now_cn.hour == 15 and now_cn.minute < 30):
                df = df[df["日期"].dt.normalize() < today].reset_index(drop=True)
                if df.empty:
                    raise RuntimeError(f"{symbol} 过滤未完成的当日日线后没有可用数据；请在收盘后重试")
            return df
        except Exception as e:
            last_error = e
            if attempt < 4:
                print(f"  {symbol} 第{attempt}次失败（{type(e).__name__}: {e}），{wait}s 后重试...")
                time.sleep(wait)
                wait += 10
    raise RuntimeError(f"{symbol} 重试4次仍失败：{type(last_error).__name__}: {last_error}")


def validate_latest_dates(latest, max_age_days=7):
    """检查数据日期是否一致/过旧。返回 (reference_date, stale_codes, warnings)。"""
    if not latest:
        return None, set(), ["没有任何品种取得有效数据"]
    dates = {code: pd.Timestamp(item["date"]).normalize() for code, item in latest.items()}
    ref_date = max(dates.values())
    warnings = []
    stale_codes = set()
    now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
    age_days = (pd.Timestamp(now_cn.date()) - ref_date).days
    if age_days > max_age_days:
        warnings.append(
            f"最新数据日期 {ref_date:%Y-%m-%d} 距中国日期 {now_cn:%Y-%m-%d} 已有 {age_days} 天；"
            "可能是接口数据过期或长假，请确认后再运行，信号已停止生成。"
        )
        stale_codes.update(latest.keys())
    for code, date in dates.items():
        lag = (ref_date - date).days
        if lag >= 2:
            warnings.append(f"{code} 数据日期 {date:%Y-%m-%d} 落后于最新日期 {ref_date:%Y-%m-%d} 达 {lag} 天，该品种不参与信号判断。")
            stale_codes.add(code)
        elif lag == 1:
            warnings.append(f"{code} 数据日期 {date:%Y-%m-%d} 比最新日期晚1天（接口更新延迟），正常使用。")
    return ref_date, stale_codes, warnings


def prep(df, mult, fee, name):
    o, h, l, c = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(c)
    tr = np.zeros(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = pd.Series(tr).rolling(ATR_N).mean().to_numpy()
    entry_hi = pd.Series(h).rolling(ENTRY_N).max().shift(1).to_numpy()
    entry_lo = pd.Series(l).rolling(ENTRY_N).min().shift(1).to_numpy()
    exit_hi = pd.Series(h).rolling(EXIT_N).max().shift(1).to_numpy()
    exit_lo = pd.Series(l).rolling(EXIT_N).min().shift(1).to_numpy()
    dates = df["日期"].to_numpy()
    return dict(name=name, mult=mult, fee=fee, dates=dates,
                open=o, high=h, low=l, close=c, atr=atr,
                entry_hi=entry_hi, entry_lo=entry_lo, exit_hi=exit_hi, exit_lo=exit_lo)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pos_path = os.path.join(here, "positions.json")
    try:
        if os.path.exists(pos_path):
            with open(pos_path, encoding="utf-8") as f:
                positions = json.load(f)
        else:
            positions = {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"[错误] 无法读取 {pos_path}: {e}。请检查 positions.json 格式。")
        return
    if not isinstance(positions, dict):
        print("[错误] positions.json 顶层必须是 JSON 对象，例如 {}。")
        return
    for code, pos in positions.items():
        if code not in FUTURES:
            print(f"[警告] positions.json 中有未知品种 {code}，该持仓不会被正常估值。")
        if not isinstance(pos, dict):
            print(f"[错误] positions.json 的 {code} 必须是对象。")
            return
        if pos.get("side", 0) not in (-1, 0, 1):
            print(f"[错误] {code} 的 side 必须是 -1、0 或 1。")
            return

    codes = ACTIVE if ACTIVE else list(FUTURES.keys())
    print("拉取最新日线并计算指标...")
    prepared = {}
    failed = []
    for code in codes:
        name, mult, margin, fee = FUTURES[code]
        try:
            df = fetch_futures(code)
            p = prep(df, mult, fee, name)
        except Exception as e:
            failed.append((name, code, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}({code}): 获取失败，跳过（{type(e).__name__}）")
            continue
        if len(p["close"]) < ENTRY_N + 10:
            print(f"  {name}: 数据不足，跳过"); continue
        prepared[code] = p

    if failed:
        print(f"\n[警告] {len(failed)} 个品种获取失败，未参与信号判断：")
        for name, code, err in failed:
            print(f"    {name}({code}): {err}")

    # 最新一行
    latest = {}
    for code, pr in prepared.items():
        i = len(pr["close"]) - 1
        latest[code] = dict(
            code=code, name=pr["name"], mult=pr["mult"], fee=pr["fee"], date=pr["dates"][i],
            open=float(pr["open"][i]), high=float(pr["high"][i]), low=float(pr["low"][i]),
            close=float(pr["close"][i]), atr=float(pr["atr"][i]),
            entry_hi=float(pr["entry_hi"][i]), entry_lo=float(pr["entry_lo"][i]),
            exit_hi=float(pr["exit_hi"][i]), exit_lo=float(pr["exit_lo"][i]))

    if not latest:
        print("\n[严重错误] 所有品种均未取得有效日线，停止生成信号。请检查网络、AKShare/Sina 接口和数据源状态。")
        return
    ref_date, stale_codes, date_warnings = validate_latest_dates(latest)
    dstr = str(pd.Timestamp(ref_date))[:10]
    if date_warnings:
        print("\n[数据日期检查]")
        for warning in date_warnings:
            print("  [警告] " + warning)
    if stale_codes:
        print("[安全保护] 日期过旧或落后的品种不会生成新开仓信号。")

    # 当前持仓数（positions 中 code 有效且 side!=0）
    npos = sum(1 for c in positions if c in latest and positions[c].get("side", 0) != 0)
    margin_cap = EQUITY * 0.80

    print("\n" + "=" * 58)
    print("郑商所杯 · 明日操作指令")
    print(f"数据截止：{dstr} ｜ 账户 {EQUITY} ｜ 单笔风险{RISK_PCT:.1%} ｜ 最多{MAX_POS}仓 ｜ 当前持仓 {npos}")
    print("=" * 58)

    # ---------- 2. 持仓处理：拆成 明日平仓 / 继续持有 两组 ----------
    exits, holds, stale_holds = [], [], []
    for code, pos in list(positions.items()):
        if code not in latest or pos.get("side", 0) == 0:
            continue
        if code in stale_codes:
            stale_holds.append((code, FUTURES.get(code, (code,))[0]))
            continue
        L = latest[code]; side = pos["side"]; entry = pos["entry"]; units = pos.get("units", 1)
        stop = pos.get("stop")
        if not stop:
            stop = entry - STOP_K * L["atr"] if side == 1 else entry + STOP_K * L["atr"]
        pnl = side * (L["close"] - entry) * L["mult"] * units
        d = "多" if side == 1 else "空"
        # 20日通道退出（收盘确认）
        exit_sig = (side == 1 and L["close"] < L["exit_lo"]) or (side == -1 and L["close"] > L["exit_hi"])
        row = dict(code=code, name=L["name"], side=side, units=units, entry=entry,
                   stop=stop, close=L["close"], pnl=pnl, d=d,
                   entry_hi=L["entry_hi"], entry_lo=L["entry_lo"],
                   exit_hi=L["exit_hi"], exit_lo=L["exit_lo"])
        if exit_sig:
            row["reason"] = f"收盘{L['close']:.0f}已{'破前20日低' if side==1 else '破前20日高'}，20日通道退出"
            exits.append(row)
        else:
            holds.append(row)

    def _fmt(x):
        return f"{x:.0f}" if np.isfinite(x) else "--"

    def chan_line(r):
        # 55日入场通道、20日退出通道；空单退出看20日高，多单退出看20日低
        exit_ref = r["exit_hi"] if r["side"] == -1 else r["exit_lo"]
        return (f"      55日通道 {_fmt(r['entry_lo'])} ~ {_fmt(r['entry_hi'])} ｜ "
                f"20日通道 {_fmt(r['exit_lo'])} ~ {_fmt(r['exit_hi'])} ｜ "
                f"{'空单退出线20日高' if r['side']==-1 else '多单退出线20日低'} {_fmt(exit_ref)}")

    # 2a. 明日平仓（置顶，最醒目）
    print("\n────── 【明日平仓】（20日通道退出，次日开盘执行）──────")
    if exits:
        for r in exits:
            close_dir = "买平" if r["side"] == -1 else "卖平"
            print(f"⚠ {r['code']}  {r['name']:<4} {r['d']}  {r['units']}手  开仓{r['entry']:.0f}  "
                  f"收盘{r['close']:.0f}  浮盈{r['pnl']:+,.0f}  → 次日开盘{close_dir}{r['units']}手  ｜ {r['reason']}")
            print(chan_line(r))
    else:
        print("无持仓触发20日通道退出，明日不平仓。")

    # 2b. 继续持有
    print("\n────── 【继续持有】──────")
    if holds:
        for r in holds:
            print(f"{r['code']}  {r['name']:<4} {r['d']}  {r['units']}手  开仓{r['entry']:.0f}  "
                  f"止损{r['stop']:.0f}  现价{r['close']:.0f}  浮盈{r['pnl']:+,.0f}  → 盘中触及止损 {r['stop']:.0f} 即平（{r['d']}单）")
            print(chan_line(r))
    else:
        print("当前无持仓。")

    if stale_holds:
        print("\n⚠ 数据日期异常、无法判断的持仓（请人工核对行情）：")
        for code, name in stale_holds:
            print(f"  {code} {name}")

    # ---------- 1. 明日开仓 ----------
    print("\n────── 【明日开仓】（次日开盘价成交）──────")
    sigs = []
    for code, L in latest.items():
        if code in stale_codes:
            continue
        if code in positions and positions[code].get("side", 0) != 0:
            continue  # 已有持仓不重复开
        close, atr = L["close"], L["atr"]
        if not np.isfinite(atr) or atr <= 0:
            continue
        side = 0
        if close > L["entry_hi"]:
            side = 1
        elif (not LONG_ONLY) and close < L["entry_lo"]:
            side = -1
        if side == 0:
            continue
        u = int(EQUITY * RISK_PCT / (STOP_K * atr * L["mult"]))
        if u < 1:
            continue
        sigs.append((atr * L["mult"], code, side, u))

    sigs.sort(key=lambda x: -x[0])
    opened = 0
    ledger_new = []
    for pri, code, side, u in sigs:
        if npos + opened >= MAX_POS:
            break
        L = latest[code]
        # 保证金预算
        need = u * L["mult"] * L["close"] * FUTURES[code][2]
        used = sum(positions[c]["units"] * latest[c]["mult"] * latest[c]["close"] * FUTURES[c][2]
                   for c in positions if c in latest and positions[c].get("side", 0) != 0)
        if used + need > margin_cap:
            continue
        d = "多" if side == 1 else "空"
        ref_stop = L["close"] - STOP_K * L["atr"] if side == 1 else L["close"] + STOP_K * L["atr"]
        why = f"收盘{L['close']:.0f}{'>前55日高' if side==1 else '<前55日低'}{L['entry_hi'] if side==1 else L['entry_lo']:.0f}，突破做{d}"
        print(f"{code}  {L['name']:<4} {d}  {u}手  参考入场{L['close']:.0f}  参考止损{ref_stop:.0f}  （{why}）")
        print(f"      → 次日开盘按实际开盘价成交，止损=开盘价{'-' if side==1 else '+'}2×ATR({2*L['atr']:.0f})，不足1手放弃")
        ledger_new.append({
            "signal_date": dstr, "code": code, "name": L["name"],
            "side": side, "units": u, "atr": round(float(L["atr"]), 2),
            "ref_entry": float(L["close"]), "ref_stop": float(ref_stop),
            "actual_fill": None, "final_stop": None, "status": "pending",
        })
        opened += 1

    # 写入 signal_ledger.json（同一 signal_date+code 去重）
    if ledger_new:
        ledger_path = os.path.join(here, "signal_ledger.json")
        try:
            ledger = []
            if os.path.exists(ledger_path):
                with open(ledger_path, encoding="utf-8") as f:
                    ledger = json.load(f)
            existing = {(r.get("signal_date"), r.get("code")) for r in ledger}
            for rec in ledger_new:
                if (rec["signal_date"], rec["code"]) not in existing:
                    ledger.append(rec)
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(ledger, f, ensure_ascii=False, indent=2)
            print(f"      📝 已记录 {len(ledger_new)} 条信号到 signal_ledger.json")
        except Exception as e:
            print(f"      [警告] signal_ledger.json 写入失败：{e}")
    if not sigs:
        if stale_codes:
            print("没有可执行的新开仓候选（部分/全部数据因日期异常被安全过滤，或未触发突破）。")
        else:
            print("今日收盘无品种突破前55日通道 → 明日不开新仓。")

    # ---------- 3. 空仓观望 ----------
    print("\n────── 【空仓观望】──────")
    watched = [L["name"] for code, L in latest.items()
               if not (code in positions and positions[code].get("side", 0) != 0)]
    print("未触发信号的品种：", "、".join(watched), "→ 明日不操作。")

    print("\n" + "=" * 58)
    print("提醒：手数不足1手不交易；止损盘中触及即平；不要临时加仓/梭哈。")
    print("=" * 58)


def get_machine_signals(positions=None):
    """给 run_daily.py 用：返回结构化的机器信号，不打印。
    复刻 main() 的信号判定（突破55日通道→多/空；持仓中→按持仓方向；其余→无）。
    返回 dict:
      date     : 数据截止日字符串
      failed   : 拉数失败的 [(name, code, err)]
      per_code : {code: {"name","side"(1/-1/0),"reason"}}  每个品种的机器方向
      holds    : [{code,name,side,units,stop,entry,exit_sig,close}]  当前持仓处理
      signals  : [{code,name,side,units,atr,close,stop_ref}]  明日开仓候选
      watched  : [name,...]  空仓观望品种
    """
    positions = positions or {}
    codes = ACTIVE if ACTIVE else list(FUTURES.keys())
    prepared, failed = {}, []
    for code in codes:
        name, mult, margin, fee = FUTURES[code]
        try:
            df = fetch_futures(code)
            p = prep(df, mult, fee, name)
        except Exception as e:
            failed.append((name, code, f"{type(e).__name__}: {e}"))
            continue
        if len(p["close"]) < ENTRY_N + 10:
            continue
        prepared[code] = p

    latest = {}
    for code, pr in prepared.items():
        i = len(pr["close"]) - 1
        latest[code] = dict(code=code, name=pr["name"], mult=pr["mult"], fee=pr["fee"],
                            date=pr["dates"][i], open=float(pr["open"][i]),
                            high=float(pr["high"][i]), low=float(pr["low"][i]),
                            close=float(pr["close"][i]), atr=float(pr["atr"][i]),
                            entry_hi=float(pr["entry_hi"][i]), entry_lo=float(pr["entry_lo"][i]),
                            exit_hi=float(pr["exit_hi"][i]), exit_lo=float(pr["exit_lo"][i]))

    ref_date, stale_codes, date_warnings = validate_latest_dates(latest)
    dstr = str(pd.Timestamp(ref_date))[:10] if ref_date is not None else ""
    if not latest:
        return dict(date="", failed=failed, per_code={}, holds=[], signals=[], watched=[],
                    warnings=["所有品种均未取得有效日线，未生成信号。"])

    # 每个品种的机器方向：持仓中→持仓方向；空仓→突破信号；否则无
    per_code = {}
    for code, L in latest.items():
        pos = positions.get(code) or {}
        if code in stale_codes:
            per_code[code] = dict(name=L["name"], side=0, reason="数据日期异常，已跳过")
            continue
        if pos.get("side", 0) != 0:
            per_code[code] = dict(name=L["name"], side=int(pos["side"]), reason="持仓中")
            continue
        side = 0
        close, atr = L["close"], L["atr"]
        if np.isfinite(atr) and atr > 0:
            if close > L["entry_hi"]:
                side = 1
            elif (not LONG_ONLY) and close < L["entry_lo"]:
                side = -1
        reason = "突破做多" if side == 1 else ("突破做空" if side == -1 else "无信号")
        per_code[code] = dict(name=L["name"], side=side, reason=reason)

    # 当前持仓处理
    holds = []
    for code, pos in (positions or {}).items():
        if code not in latest or pos.get("side", 0) == 0:
            continue
        if code in stale_codes:
            continue
        L = latest[code]; side = int(pos["side"]); entry = pos["entry"]; units = pos.get("units", 1)
        stop = pos.get("stop")
        if not stop:
            stop = entry - STOP_K * L["atr"] if side == 1 else entry + STOP_K * L["atr"]
        exit_sig = (side == 1 and L["close"] < L["exit_lo"]) or (side == -1 and L["close"] > L["exit_hi"])
        holds.append(dict(code=code, name=L["name"], side=side, units=units, stop=stop,
                          entry=entry, exit_sig=bool(exit_sig), close=L["close"],
                          entry_hi=L["entry_hi"], entry_lo=L["entry_lo"],
                          exit_hi=L["exit_hi"], exit_lo=L["exit_lo"]))

    # 明日开仓候选（空仓且有突破信号的品种）
    signals = []
    for code, L in latest.items():
        # 只把真正的空仓突破放入 signals；持仓方向绝不能被误报为新开仓。
        pos = (positions or {}).get(code) or {}
        if code in stale_codes or pos.get("side", 0) != 0:
            continue
        side = per_code[code]["side"]
        if side == 0:
            continue
        atr = L["atr"]
        u = int(EQUITY * RISK_PCT / (STOP_K * atr * L["mult"])) if atr > 0 else 0
        if not np.isfinite(atr) or atr <= 0 or u < 1:
            continue
        stop_ref = L["close"] - STOP_K * atr if side == 1 else L["close"] + STOP_K * atr
        signals.append(dict(code=code, name=L["name"], side=side, units=u,
                            atr=atr, close=L["close"], stop_ref=stop_ref))

    watched = [L["name"] for code, L in latest.items() if per_code[code]["side"] == 0]
    return dict(date=dstr, failed=failed, per_code=per_code, holds=holds,
                signals=signals, watched=watched, warnings=date_warnings)


if __name__ == "__main__":
    main()
