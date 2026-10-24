# 📈 CTA Trading Agent & Risk Management System 
> Lightweight Quantitative Investment Research Agent (Commodity Futures) built by Python.
> 轻量化商品期货 CTA 投研辅助 Agent，基于改良海龟法则，重点保障数据真实性、风控隔离与行为金融日志。

## ✨ Project Intro
本项目是一个完整的 FinTech 投研 Agent 闭环。自动拉取郑商所 18 个主力合约行情，计算 55/20 日唐奇安通道信号，结合 2倍 ATR 动态仓位管理输出多空指令。同时引入 NLP 新闻情绪分析与多层数据校验逻辑，彻底规避量化回测中的“未来函数”与“数据延迟”等致命 Bug。

## 🎯 Core Features (核心功能)
1. **多品种截面趋势扫描**：每日自动扫描 18 个活跃期货品种，基于 55 日突破寻找趋势启动点。
2. **ATR 动态风险管理**：拒绝满仓梭哈。系统根据真实波幅（ATR）动态计算手数，严格将单笔交易最大风险控制在账户的 0.8%，最多持仓 5 个品种。
3. **时序校验模块（防未来函数）**：严格区分“收盘信号”与“次日开盘执行”，并设计了“次日开盘滑点重算止损”的闭环逻辑，确保回测与实盘完全一致。
4. **NLP 新闻情绪雷达**：通过抓取财联社快讯，调用 LLM 进行打分（-5 到 +5），辅助判断宏观情绪与盘面趋势是否共振。
5. **Human-in-the-loop (人在回路)**：设计了“信号账本（signal_ledger.json）”，强制记录“机器原始信号”与“人工实际执行”的偏差，用于行为金融学分析。

## 🛠️ Stack (技术栈)
- Python 3.9+
- Pandas / Numpy / AkShare (多源数据接口)
- LLM API (新闻情绪量化与逻辑交叉校验)

## 📌 Risk & Testing Summary (实战踩坑与修复记录)
- **Fix Look-ahead Bias**：彻底修复了“用当日收盘价做当日开仓”的作弊逻辑，改为严格次日开盘执行。
- **Data Alignment Bug**：解决了新浪主力连续合约与真实月份合约（如 SH701）价差高达 500 点的数据失真问题，强制使用真实合约行情。
- **止损口径统一**：修复了代码中因变量命名歧义导致的“4倍 ATR 止损”重大风险 Bug。
- **Delay Check**：加入了数据时效性校验（若当日数据未更新，程序自动拦截并拒绝发令）。

## 🚀 Quick Start (如何运行)
```bash
pip install -r requirements.txt
python src/signal_tool.py
python src/run_daily.py
```

📝 Notes & Disclaimer

· 本项目仅用于量化投研学习与郑商所杯模拟盘实战演练，不构成任何真实投资建议。
· 实盘记录日志（CSV/JSON）已脱敏并加入 .gitignore 保护。
