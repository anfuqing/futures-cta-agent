import time
import numpy as np
import pandas as pd
import akshare as ak
# ================= 配置 =================
EQUITY = 500000            # 模拟账户
RISK_PCT = 0.008           # 单笔风险：账户的 0.8%（ATR 资金管理核心）
FEE_MULT = 1.5             # 手续费 ×1.5
FETCH_START, BT_START, END = "20230101", "20251001", "20261017"   # 拉3年算指标，回测近一年
ENTRY_N, EXIT_N, ATR_N, STOP_K = 55, 20, 20, 2.0
FILTER_N = 100             # 可选过滤：多头收盘需在均线上方，空头收盘需在均线下方
ADX_N, ADX_THRESH = 14, 20  # ADX 参数（保留备用）
ADX_FILTER = False            # ADX 趋势过滤开关：True=ADX>阈值才开仓，False=关闭
MAX_POS = 5                # 最多同时持仓 5 个品种
LONG_ONLY = False          # 双向交易（海龟原版）

# 郑商所品种池（symbol: 名称,乘数,保证金率,单边手续费元/手）
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

# 参与交易的品种：空列表 = 全部 18 个；非空 = 只跑指定的
ACTIVE = []


# ================= 数据获取 + 后复权（抹平换月跳空，已知局限：8%阈值为启发式） =================
def back_adjust(df, thresh=0.08):
    p = df[["open", "high", "low", "close"]].astype(float).to_numpy().copy()
    for i in range(1, len(p)):
        if p[i - 1, 3] > 0 and abs(p[i, 3] / p[i - 1, 3] - 1) > thresh:
            p[i:] *= p[i - 1, 3] / p[i, 3]
    df[["open", "high", "low", "close"]] = p
    return df


def fetch_futures(symbol):
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
            df = df[(df["日期"] >= pd.to_datetime(FETCH_START)) & (df["日期"] <= pd.to_datetime(END))]
            df = df.sort_values("日期").dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
            if df.empty:
                raise RuntimeError(f"{symbol} 无有效数据")
            # 禁用后复权：back_adjust() 会缩放换月跳空后的价格，导致主力连续价
            # 与真实月份合约严重偏离（如烧碱 1187 vs 真实 1726）。统一使用真实合约价。
            return df
        except Exception as e:
            print(f"  {symbol} 第{attempt}次失败（{type(e).__name__}），{wait}s 后重试...")
            time.sleep(wait); wait += 10
    raise RuntimeError(f"{symbol} 重试4次仍失败")


# ADX 平均趋向指标（Wilder 平滑，衡量趋势强度，0~100）
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


# 每品种预计算技术指标（完整序列上算，回测起点之后切片）
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



# ================= 动态持仓回测 =================
FILTER_ON = False  # 100日均线仅作实验开关；建议先用无过滤基线

def run_dynamic(prepared, max_pos=MAX_POS):
    """
    重要口径：
    1) 收盘产生的信号，下一交易日开盘成交，避免同收盘价信号/成交的前视偏差。
    2) 每日只把实际持仓的当日价格变动记入组合净值；止损日用止损成交价计算当日变动，
       不再把整笔交易盈亏重复加到净值。
    3) 交易台账盈亏、手续费均以人民币元记录；组合 acc 是以初始权益为分母的日收益贡献。
    4) 仍使用主力连续数据，无法精确模拟真实月份合约的换月、到期及交割限制。
    """
    all_dates = sorted(set().union(*[set(p["dates"]) for p in prepared.values()]))
    acc = np.zeros(len(all_dates))
    pos = {k: 0 for k in prepared}
    entry = {k: 0.0 for k in prepared}
    entry_date = {k: None for k in prepared}
    stop = {k: np.nan for k in prepared}
    units = {k: 0 for k in prepared}
    prev_close = {k: np.nan for k in prepared}
    idx = {k: -1 for k in prepared}
    # pending order: (side, signal ATR, signal date, priority)
    pending = {}
    stats = {k: dict(pl=0.0, fee=0.0, trades=[], stops=0, ledger=[]) for k in prepared}
    max_used = 0
    margin_cap = EQUITY * 0.80  # 预留至少20%权益；比赛端保证金/风控规则优先

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

        # 次日开盘执行前一交易日收盘后生成的信号。
        # 先计算当前持仓数及保证金，避免超过持仓数和预留保证金预算。
        if pending:
            orders = sorted(pending.items(), key=lambda kv: -kv[1][3])
            for k, (side, signal_atr, signal_date, priority) in orders:
                if k not in todays or pos[k] != 0 or sum(v != 0 for v in pos.values()) >= max_pos:
                    continue
                pr = prepared[k]; i = todays[k]
                o = float(pr["open"][i])
                if not np.isfinite(o) or o <= 0 or not np.isfinite(signal_atr) or signal_atr <= 0:
                    continue
                # 手数按信号日 ATR 计算，实际入场价采用次日开盘价。
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
                pos[k] = side
                entry[k] = o
                entry_date[k] = d
                units[k] = u
                stop[k] = o - STOP_K * signal_atr if side == 1 else o + STOP_K * signal_atr
                fee = u * pr["fee"] * FEE_MULT
                acc[t] -= fee / EQUITY
                stats[k]["fee"] += fee
            pending = {}

        # 对当日有行情的品种处理持仓：盘中止损优先于收盘通道退出。
        for k, i in todays.items():
            pr = prepared[k]
            o, hi, lo, close = map(float, (pr["open"][i], pr["high"][i], pr["low"][i], pr["close"][i]))
            if pos[k] != 0:
                side = pos[k]
                base = entry[k] if entry_date[k] == d or pd.isna(prev_close[k]) else prev_close[k]
                hit = (side == 1 and lo <= stop[k]) or (side == -1 and hi >= stop[k])
                if hit:
                    # 若开盘已越过止损，按开盘价；否则按止损价。
                    xp = min(o, stop[k]) if side == 1 else max(o, stop[k])
                    if not np.isfinite(xp):
                        xp = stop[k]
                    day_pnl = side * (xp - base) * pr["mult"] * units[k]
                    acc[t] += day_pnl / EQUITY
                    gross = side * (xp - entry[k]) * pr["mult"] * units[k]
                    fee = units[k] * pr["fee"] * FEE_MULT
                    acc[t] -= fee / EQUITY
                    stats[k]["pl"] += gross
                    stats[k]["fee"] += fee
                    stats[k]["trades"].append(gross - 2 * fee)
                    stats[k]["stops"] += 1
                    stats[k]["ledger"].append(dict(
                        品种=pr["name"], 方向="多" if side == 1 else "空",
                        开仓日=entry_date[k], 开仓价=round(entry[k], 4), 平仓日=d,
                        平仓价=round(xp, 4), 手数=units[k], 毛盈亏=round(gross, 2),
                        手续费=round(2 * units[k] * pr["fee"] * FEE_MULT, 2),
                        净盈亏=round(gross - 2 * units[k] * pr["fee"] * FEE_MULT, 2),
                        原因="止损2N"))
                    pos[k], units[k], entry[k], entry_date[k], stop[k] = 0, 0, 0.0, None, np.nan
                else:
                    day_pnl = side * (close - base) * pr["mult"] * units[k]
                    acc[t] += day_pnl / EQUITY
                    # 收盘确认20日反向通道退出
                    exit_signal = (
                        side == 1 and np.isfinite(pr["exit_lo"][i]) and close < pr["exit_lo"][i]
                    ) or (
                        side == -1 and np.isfinite(pr["exit_hi"][i]) and close > pr["exit_hi"][i]
                    )
                    if exit_signal:
                        gross = side * (close - entry[k]) * pr["mult"] * units[k]
                        fee = units[k] * pr["fee"] * FEE_MULT
                        acc[t] -= fee / EQUITY
                        stats[k]["pl"] += gross
                        stats[k]["fee"] += fee
                        stats[k]["trades"].append(gross - 2 * units[k] * pr["fee"] * FEE_MULT)
                        stats[k]["ledger"].append(dict(
                            品种=pr["name"], 方向="多" if side == 1 else "空",
                            开仓日=entry_date[k], 开仓价=round(entry[k], 4), 平仓日=d,
                            平仓价=round(close, 4), 手数=units[k], 毛盈亏=round(gross, 2),
                            手续费=round(2 * units[k] * pr["fee"] * FEE_MULT, 2),
                            净盈亏=round(gross - 2 * units[k] * pr["fee"] * FEE_MULT, 2),
                            原因="20日通道"))
                        pos[k], units[k], entry[k], entry_date[k], stop[k] = 0, 0, 0.0, None, np.nan

            prev_close[k] = close

        # 在收盘后生成信号，下一根有效日线开盘成交。只对当前空仓品种生成。
        signals = []
        for k, i in todays.items():
            if pos[k] != 0:
                continue
            pr = prepared[k]
            close = float(pr["close"][i])
            atr = float(pr["atr"][i])
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                continue
            strong = (not ADX_FILTER) or (
                np.isfinite(pr["adx"][i]) and pr["adx"][i] > ADX_THRESH
            )
            filt_ok_long = (not FILTER_ON) or (
                np.isfinite(pr["filt"][i]) and close > pr["filt"][i]
            )
            filt_ok_short = (not FILTER_ON) or (
                np.isfinite(pr["filt"][i]) and close < pr["filt"][i]
            )
            side = 0
            if strong and np.isfinite(pr["entry_hi"][i]) and close > pr["entry_hi"][i] and filt_ok_long:
                side = 1
            elif (not LONG_ONLY) and strong and np.isfinite(pr["entry_lo"][i]) and close < pr["entry_lo"][i] and filt_ok_short:
                side = -1
            if side:
                # 按 ATR×乘数降序；这里的排序只决定同一批信号的开仓优先级。
                signals.append((k, side, atr, atr * pr["mult"]))
        signals.sort(key=lambda x: -x[3])
        for k, side, atr, priority in signals:
            pending[k] = (side, atr, d, priority)

        max_used = max(max_used, sum(v != 0 for v in pos.values()))

    # 回测结束时未平仓持仓按最后可得收盘价计入未实现盈亏，不伪造平仓交易。
    # 每日 acc 已经逐日盯市，故这里只输出未平仓清单。
    open_positions = []
    for k, side in pos.items():
        if side:
            pr = prepared[k]
            i = idx[k]
            open_positions.append(dict(
                品种=pr["name"], 方向="多" if side == 1 else "空",
                开仓日=entry_date[k], 开仓价=entry[k], 手数=units[k],
                最新收盘=float(pr["close"][i]), 止损价=stop[k]))
    return pd.Series(acc, index=all_dates), stats, max_used, open_positions


# ================= 组合绩效指标 =================
def metrics(acc):
    """根据每日收益贡献计算累计收益、年化收益、夏普比率和最大回撤。"""
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


# ================= 主流程 =================

def main():
    global FILTER_ON
    codes = ACTIVE if ACTIVE else list(FUTURES.keys())
    print(f"参与品种: {' '.join(FUTURES[c][0] for c in codes)}  （共{len(codes)}个，最大同时持仓{MAX_POS}）")
    print("拉取数据并预计算指标...")
    prepared = {}
    failed = []
    for code in codes:
        name, mult, margin, fee = FUTURES[code]
        try:
            df = fetch_futures(code)
            p = prep(df, mult, margin, fee, name)
        except Exception as e:
            failed.append((name, code, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}({code}): 获取失败，跳过（{type(e).__name__}）")
            continue
        if len(p["dates"]) < 5:
            print(f"  {name}: 回测区间数据不足，跳过"); continue
        prepared[code] = p
        print(f"  {name}: 数据 {len(p['dates'])} 天")

    if failed:
        print(f"\n[警告] {len(failed)} 个品种获取失败，未参与回测：")
        for name, code, err in failed:
            print(f"    {name}({code}): {err}")

    if len(prepared) < 2:
        raise RuntimeError("有效品种不足")

    # ---------- 单变量对照：100日均线过滤 开 / 关 ----------
    print("\n======== 单变量对照：100日均线过滤 ========")
    net_series = {}
    for f_on in [True, False]:
        FILTER_ON = f_on
        acc, stats, max_used, open_positions = run_dynamic(prepared)
        t_c, a_c, sh_c, dd_c = metrics(acc)
        tot_trades = sum(len(st["trades"]) for st in stats.values())
        wins = sum(1 for st in stats.values() for x in st["trades"] if x > 0)
        tag = "启用100均线" if f_on else "关闭100均线"
        print(f"[{tag}] 累计 {t_c*100:6.2f}%  年化 {a_c*100:6.2f}%  夏普 {sh_c:.2f}  回撤 {dd_c*100:6.2f}%  交易 {tot_trades} 笔  胜率 {wins/tot_trades*100:.0f}%" if tot_trades else f"[{tag}] 累计 {t_c*100:6.2f}%  ... 0笔")
        net_series[f_on] = (acc, stats, max_used, open_positions)

    # 用定稿设置（启用均线）做明细展示
    FILTER_ON = True
    acc, stats, max_used, open_positions = net_series[True]

    # ---------- 各品种汇总 ----------
    rows = []
    for code, st in stats.items():
        trades = st["trades"]
        net = st["pl"] - st["fee"]
        win = sum(x > 0 for x in trades) / len(trades) if trades else float("nan")
        rows.append(dict(品种=prepared[code]["name"], 净盈亏=net, 交易=len(trades),
                         胜率=win, 止损=st["stops"], 累计=st["pl"]))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n================ 各品种（定稿版：启用100均线） ================")
    print(res.to_string(index=False))

    # ---------- 组合指标 ----------
    t_c, a_c, sh_c, dd_c = metrics(acc)
    print("\n================ 组合（定稿版） ================")
    print(f"累计: {t_c*100:6.2f}%   年化: {a_c*100:6.2f}%   夏普: {sh_c:.2f}   最大回撤: {dd_c*100:6.2f}%")
    print(f"组合总交易次数: {int(res['交易'].sum())} 笔   最大同时持仓: {max_used}")
    if open_positions:
        print("\n================ 回测结束时未平仓持仓（未计为已平仓交易） ================")
        print(pd.DataFrame(open_positions).to_string(index=False))

    # ---------- 逐笔交易台账（定位回撤来源） ----------
    ledger = [r for st in stats.values() for r in st["ledger"]]
    if ledger:
        ld = pd.DataFrame(ledger).sort_values("净盈亏")
        print("\n================ 逐笔交易台账（按净盈亏升序，最亏在最前） ================")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.max_rows", 200)
        print(ld.to_string(index=False))
        print(f"\n--- 回撤归因 ---")
        neg = ld[ld["净盈亏"] < 0]
        print(f"亏损笔数: {len(neg)}/{len(ld)}   亏损合计: {neg['净盈亏'].sum():.5f}")
        print(f"最大单笔亏损: {ld['净盈亏'].min():.5f}   （{ld.iloc[0]['品种']} {ld.iloc[0]['方向']} {ld.iloc[0]['原因']} {ld.iloc[0]['开仓日']}~{ld.iloc[0]['平仓日']}）")
        # 按原因统计
        cause = ld.groupby("原因")["净盈亏"].agg(["count", "sum"]).reset_index()
        print("\n按平仓原因统计:")
        print(cause.to_string(index=False))
    else:
        print("\n无已平仓交易。")

    # ---------- 可视化：有/无100均线过滤 两组净值对比 ----------
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(12, 6))
        for f_on, (a, _, _, _) in net_series.items():
            label = "启用100均线过滤" if f_on else "关闭100均线过滤"
            ax.plot(a.index, (1 + a).cumprod(), lw=2, label=label)
        ax.set_title("单变量对照：100日均线过滤 开/关（全18品种动态持仓）")
        ax.set_ylabel("净值"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[提示] 图形界面不可用，跳过绘图（{type(e).__name__}）。回测结果已打印。")



if __name__ == "__main__":
    main()
