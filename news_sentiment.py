"""
郑商所杯 · 快讯情绪工具（news_sentiment.py）
=============================================
作用：抓取郑商所品种相关快讯 → 输出一段「复制给大模型的 prompt」→ 你把大模型给的情绪分回填到
     news_sentiment.json（{品种代码: {score, note}}），run_daily.py 读取它并入日志。

情绪分约定：-2 强烈利空 ~ +2 强烈利多，0 = 中性无影响；note 用一句话写理由。

用法（三步）：
  1. python news_sentiment.py               # 抓快讯 + 打印 prompt（整段复制）
  2. 把 prompt 发给大模型，把返回的分数按 json 结构回填进 news_sentiment.json
  3. python run_daily.py                     # 日志闭环读取该 json

【快讯来源】
  优先读同目录 news_raw.txt（快讯，每行一条，最可控）；
  否则尝试 AkShare 全球财经快讯（东财），按品种关键词过滤。
=============================================
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import akshare as ak

HERE = os.path.dirname(os.path.abspath(__file__))
SENTI_JSON = os.path.join(HERE, "news_sentiment.json")
RAW_FILE = os.path.join(HERE, "news_raw.txt")

FUTURES = {
    "SA0": "纯碱", "FG0": "玻璃", "MA0": "甲醇", "TA0": "PTA", "RM0": "菜粕",
    "SR0": "白糖", "CF0": "棉花", "AP0": "苹果", "UR0": "尿素", "OI0": "菜油",
    "PF0": "短纤", "CJ0": "红枣", "PK0": "花生", "SF0": "硅铁", "SM0": "锰硅",
    "CY0": "棉纱", "SH0": "烧碱", "PX0": "对二甲苯",
}
NAMES = set(FUTURES.values())
KEYWORDS = list(NAMES) + ["郑商所", "期货", "PX"]


def fetch_news():
    """读同目录 news_raw.txt（快讯，每行一条）；不存在则返回 []。
    说明：akshare 自带新闻接口（生意社/财联社/东财）在本机版本要么 404 要么卡重试，
    故不自动调用，改由你手动从 财联社电报/金十/东财期货/郑商所要闻 复制到 news_raw.txt。
    """
    if os.path.exists(RAW_FILE):
        try:
            with open(RAW_FILE, "r", encoding="utf-8-sig") as f:
                raw = f.read().splitlines()
        except UnicodeDecodeError:
            with open(RAW_FILE, "r", encoding="gbk") as f:
                raw = f.read().splitlines()
        lines = [ln.strip() for ln in raw if ln.strip() and not ln.strip().lstrip("﻿").startswith("#")]
        if lines:
            return lines
    return []


def build_prompt(news):
    lines = "\n".join(f"- {n}" for n in news)
    return (
        "你是郑商所期货品种的基本面快讯情绪分析助手。下面是一批今日快讯（可能含多条、也可能为空）。\n"
        "请对以下 18 个品种逐个给出情绪分：-2 强烈利空 / -1 偏空 / 0 中性无影响 / +1 偏多 / +2 强烈利多，"
        "并用一句中文写理由（没有相关快讯就 score=0、note='无相关快讯'）。只输出 JSON，不要多余文字。\n"
        "品种代码列表：SA0 纯碱, FG0 玻璃, MA0 甲醇, TA0 PTA, RM0 菜粕, SR0 白糖, CF0 棉花, AP0 苹果, "
        "UR0 尿素, OI0 菜油, PF0 短纤, CJ0 红枣, PK0 花生, SF0 硅铁, SM0 锰硅, CY0 棉纱, SH0 烧碱, PX0 对二甲苯\n\n"
        "输出 JSON 格式：{\"SA0\": {\"score\": 0, \"note\": \"...\"}, ...}\n\n"
        "今日快讯如下：\n" + (lines if lines else "（无快讯，全部按中性处理）")
    )


def ensure_scaffold():
    """在 news_sentiment.json 里给每个品种建 score/note 占位；保留已回填的值。"""
    data = {}
    if os.path.exists(SENTI_JSON):
        try:
            with open(SENTI_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    changed = False
    for code, name in FUTURES.items():
        if code not in data or not isinstance(data[code], dict):
            data[code] = {"name": name, "score": None, "note": ""}
            changed = True
        else:
            data[code].setdefault("name", name)
            data[code].setdefault("score", None)
            data[code].setdefault("note", "")
    if changed:
        with open(SENTI_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def main():
    news = fetch_news()
    ensure_scaffold()
    print("=" * 70)
    print(f"快讯情绪工具 ｜ 时间 {datetime.datetime.now():%Y-%m-%d %H:%M} ｜ 共 {len(news)} 条相关快讯")
    print("=" * 70)
    if news:
        print("\n【今日相关快讯】")
        for n in news[:30]:
            print(" -", n)
    else:
        print("\n[提示] 未抓到相关快讯。可选：把快讯手动粘贴到同目录 news_raw.txt（每行一条），再重跑。")
    print("\n" + "=" * 70)
    print("【复制下面整段给大模型】\n")
    print(build_prompt(news))
    print("\n" + "=" * 70)
    print(f"拿到结果后，把大模型返回的 JSON 分数回填到：\n  {SENTI_JSON}")
    print("（保持 {\"品种代码\": {\"score\": -2~+2, \"note\": \"理由\"}} 结构，然后跑 run_daily.py）")


if __name__ == "__main__":
    main()
