"""
郑商所杯 · 执行日志闭环（run_daily.py）
=============================================
顺序：读 positions.json → 机器信号(signal_tool) → 雷达状态(market_radar) → 情绪(news_sentiment.json)
     → 把每个品种一行追加到 上帝视角日志.csv → 生成 当日报告_YYYY-MM-DD.md

机器自动填：日志 CSV 的「机器信号」列 + md 的机器汇总部分；
人工留痕  ：日志 CSV 的「实际成交价/手数/止损价/主观干预记录/原因」由你在 CSV 里补，
             md 里的「实际成交价/止损挂单状态/宏观判断/外部观察者批注」由你在 md 里补。

所有人工干预都必须在日志里记录「剔除了谁 + 理由」，避免变成黑箱。
=============================================
"""
import os
import io
import json
import datetime
import csv
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_CSV = os.path.join(HERE, "上帝视角日志.csv")
POS_JSON = os.path.join(HERE, "positions.json")
SENTI_JSON = os.path.join(HERE, "news_sentiment.json")

HEADERS = ["日期", "品种", "机器信号", "实际成交价", "手数", "止损价", "主观干预记录", "原因"]

import signal_tool
import market_radar

SIDE = {1: "多", -1: "空", 0: "无"}


def load_positions():
    if os.path.exists(POS_JSON):
        with io.open(POS_JSON, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    return {}


def load_sentiment():
    if not os.path.exists(SENTI_JSON):
        return {}
    try:
        with io.open(SENTI_JSON, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"[提示] 读取 news_sentiment.json 失败（{type(e).__name__}），按无情绪处理。")
        return {}


def read_existing():
    """读回日志 CSV 已有行，返回 (header_ok, set_of_(日期,品种), 旧行列表)。"""
    if not os.path.exists(LOG_CSV):
        return True, set(), []
    pairs, rows = set(), []
    with io.open(LOG_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r or not r[0]:
                continue
            rows.append(r)
            if len(r) >= 2:
                pairs.add((r[0], r[1]))
    return True, pairs, rows


def append_log(date_str, machine, signals_by_code, holds_by_code):
    """把每个品种一行追加进日志 CSV（同一日期+品种已存在则跳过）。"""
    header_ok, pairs, rows = read_existing()
    new_rows = []
    for code, m in machine.items():
        key = (date_str, f"{m['name']}{code}")
        if key in pairs:
            continue
        side = m["side"]
        fill, units, stop = "", "", ""
        if code in holds_by_code:
            h = holds_by_code[code]
            fill, units, stop = h["entry"], h["units"], h["stop"]
        elif code in signals_by_code:
            s = signals_by_code[code]
            units, stop = s["units"], s["stop_ref"]
        rows.append([date_str, f"{m['name']}{code}", SIDE[side], fill, units, stop, "", ""])
        new_rows.append(f"{m['name']}{code}")
    with io.open(LOG_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not header_ok:
            writer.writerow(HEADERS)
        for r in rows:
            writer.writerow(r)
    return new_rows


def write_report(date_str, dstr, sig, radar, senti, positions):
    """生成当日 md 报告，机器部分自动填，人工部分留空待补。"""
    today = date_str
    lines = []
    lines.append(f"# 郑商所杯 · 当日报告 {today}")
    lines.append("")
    lines.append(f"> 数据截止：{dstr} ｜ 账户 500,000 ｜ 生成：run_daily.py")
    lines.append("")
    lines.append("## 一、机器信号汇总")
    lines.append("")
    lines.append("### 明日开仓候选（次日开盘价成交，最多 5 仓，波动大者优先）")
    if sig["signals"]:
        for s in sig["signals"]:
            lines.append(f"- {s['code']} {s['name']}  {SIDE[s['side']]}  {s['units']}手  "
                         f"参考入场{s['close']:.0f}  参考止损{s['stop_ref']:.0f}")
    else:
        lines.append("- 今日收盘无品种突破前 55 日通道 → 明日不开新仓。")
    lines.append("")
    def _f(x):
        return f"{x:.0f}" if np.isfinite(x) else "--"

    def chan_str(h):
        exit_ref = h["exit_hi"] if h["side"] == -1 else h["exit_lo"]
        exit_lab = "20日高" if h["side"] == -1 else "20日低"
        return (f"55日 {_f(h['entry_lo'])}~{_f(h['entry_hi'])} ｜ "
                f"20日 {_f(h['exit_lo'])}~{_f(h['exit_hi'])} ｜ 退出线{exit_lab} {_f(exit_ref)}")

    exit_holds = [h for h in sig["holds"] if h["exit_sig"]]
    keep_holds = [h for h in sig["holds"] if not h["exit_sig"]]
    lines.append("### 明日平仓（20日通道退出，次日开盘执行）")
    if exit_holds:
        for h in exit_holds:
            close_dir = "买平" if h["side"] == -1 else "卖平"
            lines.append(f"- ⚠️ {h['code']} {h['name']}  {SIDE[h['side']]}  {h['units']}手  "
                         f"开仓{h['entry']:.0f}  收盘{h['close']:.0f}  → 次日开盘{close_dir}{h['units']}手（20日通道退出）")
            lines.append(f"  - 通道：{chan_str(h)}")
    else:
        lines.append("- 无持仓触发20日通道退出，明日不平仓。")
    lines.append("")
    lines.append("### 继续持有")
    if keep_holds:
        for h in keep_holds:
            lines.append(f"- {h['code']} {h['name']}  {SIDE[h['side']]}  {h['units']}手  "
                         f"开仓{h['entry']:.0f}  止损{h['stop']:.0f}  → 持有，盘中触及止损 {h['stop']:.0f} 即平")
            lines.append(f"  - 通道：{chan_str(h)}")
    else:
        lines.append("- 当前无持仓。")
    lines.append("")
    lines.append("### 空仓观望")
    lines.append("- " + ("、".join(sig["watched"]) if sig["watched"] else "（无）"))
    if sig["failed"]:
        lines.append("")
        lines.append(f"⚠️ 拉数失败 {len(sig['failed'])} 个品种，未参与信号："
                     + "；".join(f"{n}({c})" for n, c, _ in sig["failed"]))
    lines.append("")
    lines.append("## 二、雷达状态（参考，不自动删信号）")
    lines.append("")
    lines.append("| 代码 | 品种 | ADX | 状态 | 可跟 |")
    lines.append("|---|---|---|---|---|")
    for r in radar:
        lines.append(f"| {r['代码']} | {r['品种']} | {r['ADX']} | {r['状态']} | {'是' if r['可跟'] else '否'} |")
    lines.append("")
    lines.append("## 三、快讯情绪（news_sentiment.json，0 表示中性/未填）")
    lines.append("")
    lines.append("| 代码 | 品种 | 情绪分 | 理由 |")
    lines.append("|---|---|---|---|")
    any_senti = False
    for code, m in sig["per_code"].items():
        s = senti.get(code)
        if s and isinstance(s, dict) and s.get("score") is not None:
            any_senti = True
            lines.append(f"| {code} | {m['name']} | {s['score']} | {s.get('note','')} |")
    if not any_senti:
        lines.append("| - | - | 未回填 | 先跑 news_sentiment.py 并回填，或忽略 |")
    lines.append("")
    lines.append("## 四、人工填写区（今日必填，机器不代写）")
    lines.append("")
    lines.append("### 1) 明日实际成交 / 当日成交留痕")
    lines.append("| 品种 | 实际成交价 | 手数 | 止损挂单状态(挂单/成交/未挂) | 备注 |")
    lines.append("|---|---|---|---|---|")
    lines.append("|  |  |  |  |  |")
    lines.append("")
    lines.append("### 2) 主观干预记录（必须写清楚：剔除了谁 + 理由）")
    lines.append("> 所有干预请在 上帝视角日志.csv 的「主观干预记录/原因」列同步补上。")
    lines.append("")
    lines.append("- 干预品种/动作：")
    lines.append("  - 原因：")
    lines.append("")
    lines.append("### 3) 宏观判断")
    lines.append("- ")
    lines.append("")
    lines.append("### 4) 外部观察者批注")
    lines.append("- ")
    lines.append("")
    lines.append("---")
    lines.append(f"*自动生成于 {datetime.datetime.now():%Y-%m-%d %H:%M}；机器输出仅作参考，成交以次日开盘为准，止损盘中触及即平。*")
    lines.append("")
    path = os.path.join(HERE, f"当日报告_{today}.md")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    today = datetime.date.today().isoformat()
    positions = load_positions()
    senti = load_sentiment()

    print("拉取机器信号（signal_tool）...")
    sig = signal_tool.get_machine_signals(positions)
    if sig["failed"]:
        print(f"[警告] {len(sig['failed'])} 个品种拉数失败：")
        for n, c, e in sig["failed"]:
            print(f"    {n}({c}): {e}")

    print("拉取雷达状态（market_radar）...")
    radar, radar_failed = market_radar.build_radar()

    # 保险过滤：已有持仓的品种不重复开仓；手数不足1手的不进信号；总仓不超5
    hold_codes = {h["code"] for h in sig["holds"]}
    sig["signals"] = [s for s in sig["signals"]
                      if s["code"] not in hold_codes and int(s.get("units", 0)) >= 1]
    # 按波动率排序（高波动优先），截断保证总持仓（现有+新开）不超过 5
    sig["signals"].sort(key=lambda s: -s.get("atr", 0))
    existing_n = len(hold_codes)
    max_new = max(0, 5 - existing_n)
    sig["signals"] = sig["signals"][:max_new]

    machine = sig["per_code"]
    signals_by_code = {s["code"]: s for s in sig["signals"]}
    holds_by_code = {h["code"]: h for h in sig["holds"]}

    added = append_log(today, machine, signals_by_code, holds_by_code)
    print(f"日志 CSV：本次新增 {len(added)} 行（{today}），已有日期+品种不重复追加。")

    report = write_report(today, sig["date"], sig, radar, senti, positions)
    print(f"当日报告已生成：{report}")
    print("\n提醒：实际成交价/止损挂单/干预留痕请在 上帝视角日志.csv 与 当日报告 md 里人工补全；")
    print("机器信号只是计划，成交一律次日开盘价，止损盘中触及即平。")


if __name__ == "__main__":
    main()
