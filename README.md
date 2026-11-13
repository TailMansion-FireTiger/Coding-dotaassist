# Dota2 决策辅助原型

基于 OpenDota 公开数据的 Dota 2 决策辅助原型：胜率估计、出装推荐、打肉山/推塔对比、活动区域提示。仅用于赛后复盘、训练与教练分析；Valve 限制在正式比赛中提供额外决策优势的第三方工具，请勿接入天梯对局。

## 安装
`pip install -r requirements.txt`

## 运行
- `python dota_assist.py fetch`  拉取 OpenDota 数据
- `python dota_assist.py train`  训练模型
- `python dota_assist.py serve`  启动服务，打开 http://127.0.0.1:8000 （加 `?demo=1` 载入示例对局）

完整架构、扩展接口、已知局限和外部审查清单见 `dota_assist.py` 文件开头的说明。
