"""
郑商所杯 · 市场状态诊断工具（market_status.py）
=============================================
作用：跑一次，输出 18 个品种当前是【强趋势（可跟）/ 弱趋势（谨慎）/ 震荡（别碰）】。
     配合 signal_tool.py 使用：主程序出信号后，用本工具过滤——只做强趋势品种，
     震荡品种哪怕有信号也跳过。

【判断标准】
  ADX ≥ 25            → 强趋势：可以跟随海龟突破信号（方向看价格相对100日均线）
  ADX 15 ~ 25         → 弱趋势：谨慎，尽量少开
  ADX < 15            → 震荡：直接过滤，不做这个品种
辅助：
  均线偏离 = (收盘/100日均线 - 1)：越大说明脱离均线越远（强趋势或过热）
  布林带宽 = (上轨-下轨)/中轨：收窄=震荡蓄势，放大=趋势可能爆发

【用法】直接运行，输出按状态分类的表格。
=============================================
"""
import time
import numpy as np
import pandas as pd
import akshare as ak

# ================= 配置 =================
FETCH_START = "20230701"   # 拉取起点（保证有足够历史算 100 日均线/ADX）
ADX_N = 14
BOLL_N = 20
BOLL_K = 2.0
FILTER_N = 100

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
            df = df[df["日期"] >= pd.to_datetime(FETCH_START)]
            df = df.sort_values("日期").dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
            if df.empty:
                raise RuntimeError(f"{symbol} 无有效数据")
            # 禁用后复权：统一使用真实合约价，避免主力连续价失真（如烧碱 1187 vs 真实 1726）。
            return df
        except Exception as e:
            print(f"  {symbol} 第{attempt}次失败（{type(e).__name__}），{wait}s 后重试...")
            time.sleep(wait); wait += 10
    raise RuntimeError(f"{symbol} 重试4次仍失败")


# ADX（Wilder 平滑）
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


def main():
    codes = ACTIVE if ACTIVE else list(FUTURES.keys())
    print("拉取最新日线并计算指标...")
    rows = []
    failed = []
    for code in codes:
        name, mult, margin, fee = FUTURES[code]
        try:
            df = fetch_futures(code)
        except Exception as e:
            failed.append((name, code, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}({code}): 获取失败，跳过（{type(e).__name__}）")
            continue
        c = df["close"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        if len(c) < 120:
            print(f"  {name}: 数据不足，跳过"); continue
        # 最新值
        close = float(c[-1])
        ma100 = float(pd.Series(c).rolling(FILTER_N).mean().iloc[-1])
        adx = float(adx_ind(h, l, c)[-1])
        # 布林带带宽
        mid = pd.Series(c).rolling(BOLL_N).mean()
        std = pd.Series(c).rolling(BOLL_N).std()
        up = (mid + BOLL_K * std).iloc[-1]
        dn = (mid - BOLL_K * std).iloc[-1]
        mid_now = float(mid.iloc[-1])
        bandwidth = (up - dn) / mid_now * 100 if mid_now > 0 else float("nan")
        dev = (close / ma100 - 1) * 100 if ma100 > 0 else float("nan")
        # 分类
        if adx >= 25:
            state = "强趋势·多头" if close >= ma100 else "强趋势·空头"
        elif adx >= 15:
            state = "弱趋势·谨慎"
        else:
            state = "震荡·别碰"
        rows.append(dict(代码=code, 品种=name, ADX=round(adx, 1),
                         方向="多" if close >= ma100 else "空",
                         均线偏离=round(dev, 1), 布林带宽=round(bandwidth, 1), 状态=state))

    if failed:
        print(f"\n[警告] {len(failed)} 个品种获取失败，未参与市场状态诊断：")
        for name, code, err in failed:
            print(f"    {name}({code}): {err}")

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 70)
    print("市场状态诊断（18品种）")
    print("=" * 70)
    for grp, title in [("强趋势", "【强趋势 · 可跟海龟信号】"),
                       ("弱趋势", "【弱趋势 · 谨慎，尽量少开】"),
                       ("震荡", "【震荡 · 别碰，有信号也跳过】")]:
        sub = res[res["状态"].str.startswith(grp)]
        print(f"\n{title}")
        if sub.empty:
            print("  （无）")
        else:
            print(sub.to_string(index=False))
    print("\n" + "=" * 70)
    print("解读：")
    print("  · 均线偏离>0 且 ADX≥25 = 多头强趋势；<0 = 空头强趋势")
    print("  · 布林带宽收窄（贴近该品种近期低点）≈ 震荡蓄势，别追")
    print("  · 带宽突然放大 ≈ 趋势可能启动，海龟信号更可信")
    print("  · 用法：主程序 signal_tool 出的信号，先看本工具该品种状态，")
    print("    只有【强趋势】才执行；【震荡】直接无视，等趋势。")
    print("=" * 70)


if __name__ == "__main__":
    main()
