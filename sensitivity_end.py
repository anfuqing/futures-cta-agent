import time
import numpy as np
import pandas as pd
import akshare as ak

# ================= 配置（与 test_akshare.py 修订版一致） =================
EQUITY = 500000
RISK_PCT = 0.008
FEE_MULT = 1.5
FETCH_START = "20230101"
BT_START = "20251001"
ENTRY_N, EXIT_N, ATR_N, STOP_K = 55, 20, 20, 2.0
FILTER_N = 100
ADX_N, ADX_THRESH = 14, 20
ADX_FILTER = False
MAX_POS = 5
LONG_ONLY = False
FILTER_ON = False   # 基线：不启用100均线过滤（与修订版默认一致）

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
ACTIVE = []
NAME2MULT = {name: mult for name, mult, _, _ in FUTURES.values()}

END_CANDIDATES = ["20260831", "20260915", "20260930", "20261010", "20261017"]


def back_adjust(df, thresh=0.08):
    p = df[["open", "high", "low", "close"]].astype(float).to_numpy().copy()
    for i in range(1, len(p)):
        if p[i - 1, 3] > 0 and abs(p[i, 3] / p[i - 1, 3] - 1) > thresh:
            p[i:] *= p[i - 1, 3] / p[i, 3]
    df[["open", "high", "low", "close"]] = p
    return df


def fetch_futures(symbol, end="20261017"):
    wait = 10
    for attempt in range(1, 5):
        try:
            df = ak.futures_main_sina(symbol=symbol)
            df = df.rename(columns={
                next(c for c in ["日期", "date"] if c in df.columns): "日期",
                next(c for c in ["开盘价", "open"] if c in df.columns): "open",
                next(c for c in ["最高价", "high"] if c in df.columns): "high",
                next(c for c in ["最低价", "low"] if c in df.columns): "low",
                next(c for c in ["收盘价", "close"] if c in df.columns): "close",
            })[["日期", "open", "high", "low", "close"]]
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["日期"] = pd.to_datetime(df["日期"])
            df = df[(df["日期"] >= pd.to_datetime(FETCH_START)) & (df["日期"] <= pd.to_datetime(end))]
            df = df.sort_values("日期").dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
            if df.empty:
                raise RuntimeError(f"{symbol} 无有效数据")
            # 禁用后复权：统一使用真实合约价，避免主力连续价失真（如烧碱 1187 vs 真实 1726）。
            return df
        except Exception as e:
            print(f"  {symbol} 第{attempt}次失败（{type(e).__name__}），{wait}s 后重试...")
            time.sleep(wait); wait += 10
    raise RuntimeError(f"{symbol} 重试4次仍失败")


def adx_ind(high, low, close, period=ADX_N):
    n = len(close)
    up = np.diff(high); dn = -np.diff(low)
    plus_dm = np.concatenate([[0.0], np.where((up > dn) & (up > 0), up, 0.0)])
    minus_dm = np.concatenate([[0.0], np.where((dn > up) & (dn > 0), dn, 0.0)])
    tr = np.zeros(n); tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    alpha = 1.0 / period
    tr_s = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    plus_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    minus_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    plus_di = 100 * plus_s / np.where(tr_s == 0, np.nan, tr_s)
    minus_di = 100 * minus_s / np.where(tr_s == 0, np.nan, tr_s)
    denom = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / np.where(denom == 0, np.nan, denom)
    return pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def prep(df, mult, margin, fee, name):
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
    filt = pd.Series(c).rolling(FILTER_N).mean().to_numpy()
    adx = adx_ind(h, l, c)
    dates = df["日期"].to_numpy()
    bt_ns = pd.Timestamp(BT_START).to_datetime64()
    keep = dates >= bt_ns
    return dict(name=name, mult=mult, margin=margin, fee=fee, dates=dates[keep],
                open=o[keep], high=h[keep], low=l[keep], close=c[keep],
                atr=atr[keep], entry_hi=entry_hi[keep], entry_lo=entry_lo[keep],
                exit_hi=exit_hi[keep], exit_lo=exit_lo[keep], filt=filt[keep], adx=adx[keep])


def run_dynamic(prepared, max_pos=MAX_POS):
    all_dates = sorted(set().union(*[set(p["dates"]) for p in prepared.values()]))
    acc = np.zeros(len(all_dates))
    pos = {k: 0 for k in prepared}
    entry = {k: 0.0 for k in prepared}
    entry_date = {k: None for k in prepared}
    stop = {k: np.nan for k in prepared}
    units = {k: 0 for k in prepared}
    prev_close = {k: np.nan for k in prepared}
    idx = {k: -1 for k in prepared}
    pending = {}
    stats = {k: dict(pl=0.0, fee=0.0, trades=[], stops=0, ledger=[]) for k in prepared}
    max_used = 0
    margin_cap = EQUITY * 0.80

    for t, d in enumerate(all_dates):
        todays = {}
        for k, pr in prepared.items():
            i = idx[k] + 1
            while i < len(pr["dates"]) and pr["dates"][i] <= d:
                i += 1
            i -= 1
            if i >= 0 and pr["dates"][i] == d:
                idx[k] = i
                todays[k] = i

        if pending:
            orders = sorted(pending.items(), key=lambda kv: -kv[1][3])
            for k, (side, signal_atr, signal_date, priority) in orders:
                if k not in todays or pos[k] != 0 or sum(v != 0 for v in pos.values()) >= max_pos:
                    continue
                pr = prepared[k]; i = todays[k]
                o = float(pr["open"][i])
                if not np.isfinite(o) or o <= 0 or not np.isfinite(signal_atr) or signal_atr <= 0:
                    continue
                u = int(EQUITY * RISK_PCT / (STOP_K * signal_atr * pr["mult"]))
                if u < 1:
                    continue
                used_margin = sum(
                    units[q] * prepared[q]["mult"] * entry[q] * prepared[q]["margin"]
                    for q in prepared if pos[q] != 0
                )
                need_margin = u * pr["mult"] * o * pr["margin"]
                if used_margin + need_margin > margin_cap:
                    continue
                pos[k] = side; entry[k] = o; entry_date[k] = d; units[k] = u
                stop[k] = o - STOP_K * signal_atr if side == 1 else o + STOP_K * signal_atr
                fee = u * pr["fee"] * FEE_MULT
                acc[t] -= fee / EQUITY
                stats[k]["fee"] += fee
            pending = {}

        for k, i in todays.items():
            pr = prepared[k]
            o, hi, lo, close = map(float, (pr["open"][i], pr["high"][i], pr["low"][i], pr["close"][i]))
            if pos[k] != 0:
                side = pos[k]
                base = entry[k] if entry_date[k] == d or pd.isna(prev_close[k]) else prev_close[k]
                hit = (side == 1 and lo <= stop[k]) or (side == -1 and hi >= stop[k])
                if hit:
                    xp = min(o, stop[k]) if side == 1 else max(o, stop[k])
                    if not np.isfinite(xp):
                        xp = stop[k]
                    day_pnl = side * (xp - base) * pr["mult"] * units[k]
                    acc[t] += day_pnl / EQUITY
                    gross = side * (xp - entry[k]) * pr["mult"] * units[k]
                    fee = units[k] * pr["fee"] * FEE_MULT
                    acc[t] -= fee / EQUITY
                    stats[k]["pl"] += gross; stats[k]["fee"] += fee
                    stats[k]["trades"].append(gross - 2 * fee); stats[k]["stops"] += 1
                    stats[k]["ledger"].append(dict(品种=pr["name"], 方向="多" if side == 1 else "空",
                        开仓日=entry_date[k], 开仓价=round(entry[k], 4), 平仓日=d,
                        平仓价=round(xp, 4), 手数=units[k], 毛盈亏=round(gross, 2),
                        手续费=round(2 * units[k] * pr["fee"] * FEE_MULT, 2),
                        净盈亏=round(gross - 2 * units[k] * pr["fee"] * FEE_MULT, 2), 原因="止损2N"))
                    pos[k], units[k], entry[k], entry_date[k], stop[k] = 0, 0, 0.0, None, np.nan
                else:
                    day_pnl = side * (close - base) * pr["mult"] * units[k]
                    acc[t] += day_pnl / EQUITY
                    exit_signal = (side == 1 and np.isfinite(pr["exit_lo"][i]) and close < pr["exit_lo"][i]) or (
                        side == -1 and np.isfinite(pr["exit_hi"][i]) and close > pr["exit_hi"][i])
                    if exit_signal:
                        gross = side * (close - entry[k]) * pr["mult"] * units[k]
                        fee = units[k] * pr["fee"] * FEE_MULT
                        acc[t] -= fee / EQUITY
                        stats[k]["pl"] += gross; stats[k]["fee"] += fee
                        stats[k]["trades"].append(gross - 2 * units[k] * pr["fee"] * FEE_MULT)
                        stats[k]["ledger"].append(dict(品种=pr["name"], 方向="多" if side == 1 else "空",
                            开仓日=entry_date[k], 开仓价=round(entry[k], 4), 平仓日=d,
                            平仓价=round(close, 4), 手数=units[k], 毛盈亏=round(gross, 2),
                            手续费=round(2 * units[k] * pr["fee"] * FEE_MULT, 2),
                            净盈亏=round(gross - 2 * units[k] * pr["fee"] * FEE_MULT, 2), 原因="20日通道"))
                        pos[k], units[k], entry[k], entry_date[k], stop[k] = 0, 0, 0.0, None, np.nan
            prev_close[k] = close

        signals = []
        for k, i in todays.items():
            if pos[k] != 0:
                continue
            pr = prepared[k]
            close = float(pr["close"][i]); atr = float(pr["atr"][i])
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                continue
            strong = (not ADX_FILTER) or (np.isfinite(pr["adx"][i]) and pr["adx"][i] > ADX_THRESH)
            filt_ok_long = (not FILTER_ON) or (np.isfinite(pr["filt"][i]) and close > pr["filt"][i])
            filt_ok_short = (not FILTER_ON) or (np.isfinite(pr["filt"][i]) and close < pr["filt"][i])
            side = 0
            if strong and np.isfinite(pr["entry_hi"][i]) and close > pr["entry_hi"][i] and filt_ok_long:
                side = 1
            elif (not LONG_ONLY) and strong and np.isfinite(pr["entry_lo"][i]) and close < pr["entry_lo"][i] and filt_ok_short:
                side = -1
            if side:
                signals.append((k, side, atr, atr * pr["mult"]))
        signals.sort(key=lambda x: -x[3])
        for k, side, atr, priority in signals:
            pending[k] = (side, atr, d, priority)

        max_used = max(max_used, sum(v != 0 for v in pos.values()))

    open_positions = []
    for k, side in pos.items():
        if side:
            pr = prepared[k]; i = idx[k]
            open_positions.append(dict(品种=pr["name"], 方向="多" if side == 1 else "空",
                开仓日=entry_date[k], 开仓价=entry[k], 手数=units[k],
                最新收盘=float(pr["close"][i]), 止损价=stop[k]))
    return pd.Series(acc, index=all_dates), stats, max_used, open_positions


def metrics(acc):
    acc = pd.Series(acc).dropna()
    if len(acc) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    nav = (1 + acc).cumprod()
    total = nav.iloc[-1] - 1
    n = len(acc)
    annual = (1 + total) ** (252 / max(n, 1)) - 1 if total > -1 else -1.0
    vol = acc.std() * np.sqrt(252)
    sharpe = (acc.mean() * 252) / vol if vol > 0 else float("nan")
    max_dd = (nav / nav.cummax() - 1).min()
    return total, annual, sharpe, max_dd


# ================= 一次拉全量数据，再对不同终点切片回测 =================

def main():
    codes = ACTIVE if ACTIVE else list(FUTURES.keys())
    print(f"拉取数据（到20261017）并预计算指标...")
    prepared_full = {}
    failed = []
    for code in codes:
        name, mult, margin, fee = FUTURES[code]
        try:
            df = fetch_futures(code, "20261017")
            p = prep(df, mult, margin, fee, name)
        except Exception as e:
            failed.append((name, code, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}({code}): 获取失败，跳过（{type(e).__name__}）")
            continue
        if len(p["dates"]) < 5:
            print(f"  {name}: 数据不足，跳过"); continue
        prepared_full[code] = p
        print(f"  {name}: {len(p['dates'])} 天")

    if failed:
        print(f"\n[警告] {len(failed)} 个品种获取失败，未参与敏感性分析：")
        for name, code, err in failed:
            print(f"    {name}({code}): {err}")

    print("\n======== 终点敏感性：收益对回测截止日期的依赖 ========")
    rows = []
    for end_s in END_CANDIDATES:
        end_ts = pd.Timestamp(end_s).to_datetime64()
        prepared = {}
        for code, pr in prepared_full.items():
            keep = pr["dates"] <= end_ts
            sub = {kk: (vv[keep] if isinstance(vv, np.ndarray) else vv) for kk, vv in pr.items()}
            sub["name"] = pr["name"]
            prepared[code] = sub
        acc, stats, max_used, open_pos = run_dynamic(prepared)
        t_c, a_c, sh_c, dd_c = metrics(acc)
        tot_trades = sum(len(st["trades"]) for st in stats.values())
        # 期末未平仓浮盈（元）-> 占账户比例
        unreal = 0.0
        for o in open_pos:
            mult = NAME2MULT[o["品种"]]
            side = 1 if o["方向"] == "多" else -1
            unreal += side * (o["最新收盘"] - o["开仓价"]) * mult * o["手数"]
        unreal_pct = unreal / EQUITY
        # 已实现收益 = 台账净盈亏合计 / 初始权益（NAV 累计已逐日盯市含浮盈，不在此相减）
        realized_pct = sum(r["净盈亏"] for st in stats.values() for r in st["ledger"]) / EQUITY
        rows.append(dict(终点=end_s, 累计=t_c, 夏普=sh_c, 回撤=dd_c, 交易=tot_trades,
                         期末浮盈=unreal_pct, 已实现=realized_pct))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    res["累计"] = res["累计"].map(lambda x: f"{x*100:.2f}%")
    res["夏普"] = res["夏普"].map(lambda x: f"{x:.2f}")
    res["回撤"] = res["回撤"].map(lambda x: f"{x*100:.2f}%")
    res["期末浮盈"] = res["期末浮盈"].map(lambda x: f"{x*100:.2f}%")
    res["已实现"] = res["已实现"].map(lambda x: f"{x*100:.2f}%")
    print(res.to_string(index=False))
    print("\n说明：NAV累计(复利、已逐日盯市含浮盈)、已实现(台账净盈亏/初始权益)、期末浮盈(持仓浮盈/初始权益)三者口径不同，不要求加和相等。若累计随终点大幅波动、且期末浮盈占比高，说明收益依赖回测截止时点。")



if __name__ == "__main__":
    main()
