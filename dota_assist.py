#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Dota 2 决策辅助原型  ·  dota_assist.py
================================================================================

【定位与合规】
本程序是面向复盘、训练和教练分析的离线/赛后原型，不是对局中的实时外挂。
Valve 自 2023 年起明确限制在比赛进行中提供额外决策优势的第三方工具
（例如实时出装提示、敌方位置预测）。将本工具接入天梯/比赛并在对局中展示
建议，存在违反使用条款并被封禁的风险。接收器接口仅作为结构预留，默认
不要在正式比赛中启用。

【架构概览】
  数据层 DataSource
      公开 OpenDota API（https://api.opendota.com/api/）拉取职业/已解析比赛，
      原始 JSON 缓存到 data/cache/，再次运行可完全离线。
  特征层 FeatureBuilder
      按「每场每分钟」切局面快照：时间、经济/经验差、已发生的塔/兵营/肉山
      目标计数、击杀差、双方英雄 one-hot，标签为 radiant_win。
  模型层
      WinModel            胜率模型（优先 LightGBM，失败则回退 sklearn）
      ItemRecommender     基于样本 purchase_log 的频次/胜率出装推荐
      ObjectiveAdvisor    用模型做「肉山 vs 推塔」特征 +1 的启发式对比
                          （明确不是因果推断）
      ZonePredictor       基于 lane_pos / lane_role 的粗粒度热力图提示；
                          录像逐秒坐标接口已预留（ReplayCoordinateZonePredictor）
      LLMExplainer        默认模板句；可通过环境变量挂接外部 LLM API，
                          禁止在代码中硬编码密钥
  服务层 FastAPI
      提供静态页面与 JSON 接口，并含本地状态接收桩。

【目录结构】
  dota_assist.py          本文件（全部后端）
  index.html              自包含中文前端（内联 CSS/JS）
  requirements.txt        依赖
  data/cache/             原始 API 缓存（可离线复用，体积较大可不打包）
  artifacts/              训练产物（模型、元数据、英雄/物品表、真实样本快照）

【运行方法】
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python dota_assist.py fetch          # 拉取并缓存比赛，约 150-400 场
  python dota_assist.py train          # 构图特征、训练并写出 artifacts/
  python dota_assist.py serve          # 默认 http://127.0.0.1:8000
  python dota_assist.py all            # 依次 fetch + train + serve
  常用参数：
    fetch --target 280 --sleep 1.15
    serve --host 127.0.0.1 --port 8000

【扩展接口】
  下列 Protocol / 抽象类是给后续替换实现用的稳定入口（见源码第 1 节）：
    DataSource, FeatureBuilder, WinModel, ItemRecommender,
    ObjectiveAdvisor, ZonePredictor, LLMExplainer
  挂接 LLM（可选，无需则忽略）：
    环境变量 DOTA_ASSIST_LLM_BASE_URL  例如兼容 OpenAI 的服务地址
            DOTA_ASSIST_LLM_API_KEY   密钥，切勿写入源码或提交到 git
            DOTA_ASSIST_LLM_MODEL     模型名
            DOTA_ASSIST_LLM_PATH      默认 /chat/completions
  挂接录像坐标：
    实现 ZonePredictor.predict，或继承 ReplayCoordinateZonePredictor
    并在 load_context() 中替换 zone 实例。

【已知限制】
  1. 样本来自 OpenDota 已解析比赛，偏向职业赛/主动解析局，不能代表普通天梯。
  2. 胜率模型是关联模型，对「打肉山 / 推塔」的 +1 模拟只是特征扰动启发式，
     不是反事实/因果效应，不能当作最优策略。
  3. OpenDota 的 lane_pos 基本只覆盖对线期，没有逐秒全图坐标；后期刷钱路线
     需要自行解析 .dem 录像（接口已留，本原型未实现）。
  4. 出装推荐是样本内频次与胜率，不是序列模型；冷门英雄会明显稀疏。
  5. 免费 OpenDota API 约 60 次/分钟，本程序默认间隔 1.15 秒；大批量请自行
     申请 API key 并在 OPEN_DOTA_API_KEY 中提供（可选）。
  6. 英雄中文名为内置对照，若版本出现新英雄将回退到英文 localized_name。

【供外部模型审查的清单】
  [ ] 训练/测试是否按 match_id 分组划分，有无同一场比赛泄漏到两侧？
  [ ] AUC / logloss 是否在代码中真实计算并写入 artifacts，有无写死数字？
  [ ] 目标建议文案是否清楚标明「启发式、非因果」？
  [ ] 分区提示是否标明「对线期热力图、非实时位置」？
  [ ] 是否存在把职业样本结论推广到所有分段的不当表述？
  [ ] 本地状态接收路径是否被误用为正式对局辅助？合规说明是否完整？
  [ ] 有无硬编码的 API 密钥、账号或个人隐私字段？
  [ ] 拉取数据是否遵守速率限制、是否可离线复现？
  [ ] 接口签名变更时，是否同步改了 index.html？
  [ ] 特征中的英雄 one-hot 对稀有英雄是否过于稀疏？是否需要嵌入？
  [ ] 出装过滤规则（消耗品、配方）是否合理，样本数是否始终展示？

【协议与版本】
  原型版本 0.1.0  |  数据源 OpenDota 公开 API  |  语言 Python 3.10+
================================================================================
"""

from __future__ import annotations

# =============================================================================
# 0. 导入、路径与全局常量
# =============================================================================
import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import joblib
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    lgb = None  # type: ignore
    HAS_LIGHTGBM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dota_assist")

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "cache"
CONST_DIR = CACHE_DIR / "constants"
MATCH_DIR = CACHE_DIR / "matches"
LIST_DIR = CACHE_DIR / "lists"
POP_DIR = CACHE_DIR / "item_popularity"
ART_DIR = ROOT / "artifacts"
INDEX_HTML = ROOT / "index.html"

OPEN_DOTA_BASE = os.environ.get("OPEN_DOTA_BASE", "https://api.opendota.com/api").rstrip("/")
DEFAULT_SLEEP = 1.15
DEFAULT_TARGET = 280
APP_VERSION = "0.1.0"

# 出装推荐时跳过的消耗品 / 非决策向物品（配方另外按前缀过滤）
SKIP_ITEMS = {
    "tango", "tango_single", "flask", "clarity", "faerie_fire", "enchanted_mango",
    "blood_grenade", "ward_observer", "ward_sentry", "ward_dispenser",
    "observer_and_sentry_wards", "smoke_of_deceit", "dust", "dust_of_appearance",
    "tpscroll", "tome_of_knowledge", "courier", "branches",
    "healing_lotus", "great_healing_lotus", "greater_healing_lotus",
    "famango", "great_famango", "greater_famango",
    "cheese", "aegis", "refresher_shard", "ultimate_scepter_roshan",
    "aghanims_shard", "aghanims_shard_roshan",
    "river_painter", "river_painter2", "river_painter3", "river_painter4",
    "river_painter5", "river_painter6", "river_painter7",
}

# 简体中文名；未见对照时回退英文
HERO_NAME_ZH: Dict[int, str] = {
    1: "敌法师", 2: "斧王", 3: "祸乱之源", 4: "血魔", 5: "水晶室女",
    6: "卓尔游侠", 7: "撼地者", 8: "主宰", 9: "米拉娜", 10: "变体精灵",
    11: "影魔", 12: "幻影长矛手", 13: "帕克", 14: "帕吉", 15: "剃刀",
    16: "沙王", 17: "风暴之灵", 18: "斯温", 19: "小小", 20: "复仇之魂",
    21: "风行者", 22: "宙斯", 23: "昆卡", 25: "莉娜", 26: "莱恩",
    27: "暗影萨满", 28: "斯拉达", 29: "潮汐猎人", 30: "巫医", 31: "巫妖",
    32: "力丸", 33: "谜团", 34: "修补匠", 35: "狙击手", 36: "瘟疫法师",
    37: "术士", 38: "兽王", 39: "痛苦女王", 40: "剧毒术士", 41: "虚空假面",
    42: "冥魂大帝", 43: "死亡先知", 44: "幻影刺客", 45: "帕格纳", 46: "圣堂刺客",
    47: "冥界亚龙", 48: "露娜", 49: "龙骑士", 50: "戴泽", 51: "发条技师",
    52: "拉席克", 53: "先知", 54: "噬魂鬼", 55: "黑暗贤者", 56: "克林克兹",
    57: "全能骑士", 58: "魅惑魔女", 59: "哈斯卡", 60: "暗夜魔王",
    61: "育母蜘蛛", 62: "赏金猎人", 63: "编织者", 64: "杰奇洛", 65: "蝙蝠骑士",
    66: "陈", 67: "幽鬼", 68: "远古冰魄", 69: "末日使者", 70: "熊战士",
    71: "裂魂人", 72: "矮人直升机", 73: "炼金术士", 74: "祈求者", 75: "沉默术士",
    76: "殁境神蚀者", 77: "狼人", 78: "酒仙", 79: "暗影恶魔", 80: "德鲁伊",
    81: "混沌骑士", 82: "米波", 83: "树精卫士", 84: "食人魔魔法师", 85: "不朽尸王",
    86: "拉比克", 87: "干扰者", 88: "司夜刺客", 89: "娜迦海妖", 90: "光之守卫",
    91: "艾欧", 92: "维萨吉", 93: "斯拉克", 94: "美杜莎", 95: "巨魔战将",
    96: "半人马战行者", 97: "马格纳斯", 98: "伐木机", 99: "钢背兽", 100: "巨牙海民",
    101: "天怒法师", 102: "亚巴顿", 103: "上古巨神", 104: "军团指挥官", 105: "工程师",
    106: "灰烬之灵", 107: "大地之灵", 108: "孽主", 109: "恐怖利刃", 110: "凤凰",
    111: "神谕者", 112: "寒冬飞龙", 113: "天穹守望者", 114: "齐天大圣",
    119: "邪影芳灵", 120: "石鳞剑士", 121: "天涯墨客", 123: "森海飞霞",
    126: "虚无之灵", 128: "电炎绝手", 129: "玛尔斯", 131: "百戏大王",
    135: "破晓辰星", 136: "玛西", 137: "元初野兽", 138: "琼英碧灵", 145: "凯",
    155: "拉尔戈",
}

LANE_ROLE_ZH = {1: "优势路", 2: "中路", 3: "劣势路", 4: "野区", 0: "未知"}
ZONE_ZH = {
    "radiant_base": "天辉基地",
    "dire_base": "夜魇基地",
    "mid_river": "中路/河道",
    "top_lane": "上路",
    "bot_lane": "下路",
    "radiant_jungle": "天辉野区",
    "dire_jungle": "夜魇野区",
    "roshan": "肉山附近",
}

NUMERIC_FEATURES = [
    "minute",
    "radiant_gold_adv",
    "radiant_xp_adv",
    "kills_diff",
    "radiant_towers_lost",
    "dire_towers_lost",
    "radiant_rax_lost",
    "dire_rax_lost",
    "roshan_radiant",
    "roshan_dire",
]


# =============================================================================
# 1. 扩展接口（Protocols）—— 后续替换实现时请保持这些方法签名
# =============================================================================
class DataSource(Protocol):
    """比赛与常量数据源。可替换为本地回放库或其他公开接口。"""

    def fetch_constants(self, force: bool = False) -> Dict[str, Any]: ...
    def fetch_matches(self, target: int, sleep_s: float) -> List[Path]: ...
    def iter_cached_matches(self) -> Iterable[Dict[str, Any]]: ...
    def load_heroes(self) -> List[Dict[str, Any]]: ...
    def load_items(self) -> Dict[str, Dict[str, Any]]: ...


class FeatureBuilder(Protocol):
    """把原始比赛 JSON 转成按分钟的局面快照。"""

    def build_snapshots(self, match: Mapping[str, Any]) -> List[Dict[str, Any]]: ...
    def vectorize(self, snapshots: Sequence[Mapping[str, Any]], hero_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]: ...
    def encode_query(self, query: Mapping[str, Any], hero_ids: Sequence[int], feature_names: Sequence[str]) -> np.ndarray: ...


class WinModel(Protocol):
    """局面胜率模型。predict_proba 返回天辉胜率。"""

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path, meta: Dict[str, Any]) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "WinModel": ...


class ItemRecommender(Protocol):
    def fit(self, matches: Iterable[Mapping[str, Any]]) -> Dict[str, Any]: ...
    def recommend(self, hero_id: int, minute: int, current_items: Sequence[str], top_k: int = 8) -> Any: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "ItemRecommender": ...


class ObjectiveAdvisor(Protocol):
    def advise(self, query: Mapping[str, Any], win_model: WinModel, builder: FeatureBuilder, hero_ids: Sequence[int], feature_names: Sequence[str]) -> Dict[str, Any]: ...


class ZonePredictor(Protocol):
    def fit(self, matches: Iterable[Mapping[str, Any]]) -> Dict[str, Any]: ...
    def predict(self, hero_id: int, minute: int, is_radiant: bool = True, lane_role: Optional[int] = None) -> Dict[str, Any]: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "ZonePredictor": ...


class LLMExplainer(Protocol):
    """把数值结果翻译成一两句中文建议。默认模板实现，可换外部模型。"""

    def explain(self, payload: Mapping[str, Any]) -> str: ...


# =============================================================================
# 2. 工具函数
# =============================================================================
def ensure_dirs() -> None:
    for p in (CONST_DIR, MATCH_DIR, LIST_DIR, POP_DIR, ART_DIR):
        p.mkdir(parents=True, exist_ok=True)


def hero_zh(hero_id: int, fallback: str = "") -> str:
    return HERO_NAME_ZH.get(int(hero_id), fallback or f"英雄#{hero_id}")


def is_skipped_item(key: str) -> bool:
    if not key:
        return True
    k = key.lower()
    if k.startswith("recipe_") or k.startswith("river_painter"):
        return True
    return k in SKIP_ITEMS


def time_bucket(minute: int) -> str:
    if minute < 10:
        return "0-10"
    if minute < 20:
        return "10-20"
    if minute < 30:
        return "20-30"
    if minute < 40:
        return "30-40"
    return "40+"


def cell_to_zone(x: float, y: float) -> str:
    """把 OpenDota lane_pos 格子粗分成地图区域（经验分区，非官方网格）。"""
    nx = min(1.0, max(0.0, (float(x) - 64.0) / 128.0))
    ny = min(1.0, max(0.0, (float(y) - 64.0) / 128.0))
    if nx < 0.22 and ny < 0.22:
        return "radiant_base"
    if nx > 0.78 and ny > 0.78:
        return "dire_base"
    if 0.22 <= nx <= 0.42 and 0.52 <= ny <= 0.72:
        return "roshan"
    if abs(nx - ny) < 0.14 and 0.25 < nx < 0.75:
        return "mid_river"
    if ny > 0.62 and nx < 0.75:
        return "top_lane"
    if nx > 0.62 and ny < 0.75:
        return "bot_lane"
    if nx + ny < 1.0:
        return "radiant_jungle"
    return "dire_jungle"


# =============================================================================
# 3. 数据层
# =============================================================================
class OpenDotaDataSource:
    """OpenDota 公开 API 数据源，带磁盘缓存与速率限制。"""

    def __init__(self, sleep_s: float = DEFAULT_SLEEP) -> None:
        ensure_dirs()
        self.sleep_s = sleep_s
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": f"dota-assist-prototype/{APP_VERSION} (research/coaching; offline cache)"}
        )
        api_key = os.environ.get("OPEN_DOTA_API_KEY", "").strip()
        self._params = {"api_key": api_key} if api_key else {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 5) -> Any:
        url = path if path.startswith("http") else f"{OPEN_DOTA_BASE}{path}"
        merged = dict(self._params)
        if params:
            merged.update({k: v for k, v in params.items() if v is not None})
        last_err: Optional[Exception] = None
        for i in range(retries):
            try:
                r = self.session.get(url, params=merged, timeout=30)
                if r.status_code == 429:
                    wait = 15 + i * 10
                    log.warning("速率限制 429，等待 %ss  %s", wait, url)
                    time.sleep(wait)
                    continue
                if r.status_code >= 500:
                    time.sleep(3 + i * 2)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                log.warning("请求失败 %s: %s", url, e)
                time.sleep(3 + i * 2)
        raise RuntimeError(f"请求失败: {url} ({last_err})")

    def _sleep(self) -> None:
        time.sleep(self.sleep_s)

    def fetch_constants(self, force: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in ("heroes", "items"):
            dest = CONST_DIR / f"{name}.json"
            if dest.exists() and not force:
                out[name] = json.loads(dest.read_text(encoding="utf-8"))
                continue
            log.info("拉取常量 /constants/%s", name)
            data = self._get(f"/constants/{name}")
            dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            out[name] = data
            self._sleep()
        return out

    def fetch_item_popularity(self, hero_ids: Sequence[int], force: bool = False) -> None:
        """可选：按英雄拉取阶段出装热度，供推荐回退。"""
        for hid in hero_ids:
            dest = POP_DIR / f"{hid}.json"
            if dest.exists() and not force:
                continue
            try:
                data = self._get(f"/heroes/{hid}/itemPopularity")
                dest.write_text(json.dumps(data), encoding="utf-8")
                log.info("itemPopularity hero=%s ok", hid)
            except Exception as e:
                log.warning("itemPopularity hero=%s 失败: %s", hid, e)
            self._sleep()

    def _collect_match_ids(self, target: int) -> List[int]:
        pro_all: List[Dict[str, Any]] = []
        less: Optional[int] = None
        for page in range(8):
            data = self._get("/proMatches", params={"less_than_match_id": less})
            self._sleep()
            if not data:
                break
            pro_all.extend(data)
            less = min(x["match_id"] for x in data)
            log.info("proMatches 第 %s 页 +%s，累计 %s", page + 1, len(data), len(pro_all))
            if len(pro_all) >= 700:
                break
        (LIST_DIR / "pro_matches.json").write_text(json.dumps(pro_all), encoding="utf-8")
        ids = list(dict.fromkeys(x["match_id"] for x in pro_all if x.get("version")))
        log.info("职业赛中带 version（已解析）的场次: %s / %s", len(ids), len(pro_all))

        less = None
        extra: List[int] = []
        while len(ids) + len(extra) < target + 80:
            data = self._get("/parsedMatches", params={"less_than_match_id": less})
            self._sleep()
            if not data:
                break
            batch = [x["match_id"] for x in data]
            extra.extend(batch)
            less = min(batch)
            log.info("parsedMatches +%s，补充累计 %s", len(batch), len(extra))
            if len(extra) >= 500:
                break
        (LIST_DIR / "parsed_ids.json").write_text(json.dumps(extra), encoding="utf-8")
        for mid in extra:
            if mid not in ids:
                ids.append(mid)
        log.info("候选比赛 ID: %s", len(ids))
        return ids

    @staticmethod
    def is_usable(match: Mapping[str, Any]) -> bool:
        return bool(
            match
            and match.get("radiant_gold_adv")
            and match.get("radiant_xp_adv")
            and match.get("players")
            and len(match.get("players") or []) >= 10
            and match.get("radiant_win") is not None
            and match.get("duration")
        )

    def count_usable_cached(self) -> int:
        n = 0
        for p in MATCH_DIR.glob("*.json"):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                if self.is_usable(m):
                    n += 1
            except Exception:
                continue
        return n

    def fetch_matches(self, target: int = DEFAULT_TARGET, sleep_s: Optional[float] = None) -> List[Path]:
        if sleep_s is not None:
            self.sleep_s = sleep_s
        self.fetch_constants()
        already = self.count_usable_cached()
        log.info("缓存中已有可用比赛 %s，目标 %s", already, target)
        if already >= target:
            return sorted(MATCH_DIR.glob("*.json"))
        ids = self._collect_match_ids(target)
        ok = already
        for mid in ids:
            if ok >= target:
                break
            dest = MATCH_DIR / f"{mid}.json"
            if dest.exists():
                continue
            try:
                data = self._get(f"/matches/{mid}")
            except Exception as e:
                log.warning("跳过 %s: %s", mid, e)
                self._sleep()
                continue
            self._sleep()
            if not isinstance(data, dict) or data.get("error"):
                log.warning("跳过 %s: %s", mid, data.get("error") if isinstance(data, dict) else data)
                continue
            dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if self.is_usable(data):
                ok += 1
            log.info(
                "已保存 %s usable=%s 进度 %s/%s duration=%s",
                mid, self.is_usable(data), ok, target, data.get("duration"),
            )
        log.info("拉取结束，可用比赛 %s", self.count_usable_cached())
        return sorted(MATCH_DIR.glob("*.json"))

    def iter_cached_matches(self) -> Iterable[Dict[str, Any]]:
        for p in sorted(MATCH_DIR.glob("*.json")):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if self.is_usable(m):
                yield m

    def load_heroes(self) -> List[Dict[str, Any]]:
        path = CONST_DIR / "heroes.json"
        if not path.exists():
            self.fetch_constants()
        raw = json.loads(path.read_text(encoding="utf-8"))
        heroes = []
        for _, h in raw.items():
            if not isinstance(h, dict) or "id" not in h:
                continue
            hid = int(h["id"])
            en = h.get("localized_name") or h.get("name") or str(hid)
            heroes.append({
                "id": hid,
                "name": h.get("name"),
                "localized_name": en,
                "name_zh": hero_zh(hid, en),
                "primary_attr": h.get("primary_attr"),
                "roles": h.get("roles") or [],
                "attack_type": h.get("attack_type"),
            })
        heroes.sort(key=lambda x: x["id"])
        return heroes

    def load_items(self) -> Dict[str, Dict[str, Any]]:
        path = CONST_DIR / "items.json"
        if not path.exists():
            self.fetch_constants()
        raw = json.loads(path.read_text(encoding="utf-8"))
        items: Dict[str, Dict[str, Any]] = {}
        for key, it in raw.items():
            if not isinstance(it, dict):
                continue
            dname = it.get("dname")
            if not dname:
                continue
            items[key] = {
                "key": key,
                "id": it.get("id"),
                "dname": dname,
                "cost": it.get("cost"),
                "qual": it.get("qual"),
                "components": it.get("components"),
                "skipped": is_skipped_item(key),
            }
        return items


# =============================================================================
# 4. 特征层
# =============================================================================
class SnapshotFeatureBuilder:
    """从单场已解析比赛生成每分钟快照。"""

    def build_snapshots(self, match: Mapping[str, Any]) -> List[Dict[str, Any]]:
        gold_adv = list(match.get("radiant_gold_adv") or [])
        xp_adv = list(match.get("radiant_xp_adv") or [])
        if not gold_adv or not xp_adv:
            return []
        n = min(len(gold_adv), len(xp_adv), int(match.get("duration", 0)) // 60 + 1)
        if n < 5:
            return []
        players = list(match.get("players") or [])
        rad_heroes, dire_heroes = [], []
        kill_events: List[Tuple[int, int]] = []
        for p in players:
            hid = p.get("hero_id")
            if not hid:
                return []
            is_rad = bool(p.get("isRadiant") if p.get("isRadiant") is not None else int(p.get("player_slot", 0)) < 128)
            (rad_heroes if is_rad else dire_heroes).append(int(hid))
            sign = 1 if is_rad else -1
            for ev in p.get("kills_log") or []:
                try:
                    kill_events.append((int(ev.get("time", 0)), sign))
                except Exception:
                    continue
        if len(rad_heroes) != 5 or len(dire_heroes) != 5:
            return []

        obj_events = []
        for o in match.get("objectives") or []:
            t = o.get("time")
            typ = o.get("type") or ""
            key = o.get("key") or ""
            if t is None:
                continue
            try:
                t = int(t)
            except Exception:
                continue
            if typ == "building_kill":
                side = "radiant" if "goodguys" in key else ("dire" if "badguys" in key else None)
                if side is None:
                    continue
                if "tower" in key:
                    obj_events.append((t, "tower", side))
                elif "rax" in key:
                    obj_events.append((t, "rax", side))
            elif typ == "CHAT_MESSAGE_ROSHAN_KILL":
                team = o.get("team")
                side = "radiant" if team == 2 else ("dire" if team == 3 else None)
                if side:
                    obj_events.append((t, "roshan", side))

        kill_events.sort()
        obj_events.sort()
        label = 1 if match.get("radiant_win") else 0
        mid = match.get("match_id")
        out: List[Dict[str, Any]] = []
        k_i = o_i = 0
        kd = 0
        rt = dt = rr = dr = rros = dros = 0
        for minute in range(n):
            t_end = minute * 60
            while k_i < len(kill_events) and kill_events[k_i][0] <= t_end:
                kd += kill_events[k_i][1]
                k_i += 1
            while o_i < len(obj_events) and obj_events[o_i][0] <= t_end:
                _, kind, side = obj_events[o_i]
                if kind == "tower" and side == "radiant":
                    rt += 1
                elif kind == "tower" and side == "dire":
                    dt += 1
                elif kind == "rax" and side == "radiant":
                    rr += 1
                elif kind == "rax" and side == "dire":
                    dr += 1
                elif kind == "roshan" and side == "radiant":
                    rros += 1
                elif kind == "roshan" and side == "dire":
                    dros += 1
                o_i += 1
            out.append({
                "match_id": mid,
                "minute": minute,
                "radiant_gold_adv": float(gold_adv[minute]),
                "radiant_xp_adv": float(xp_adv[minute]),
                "kills_diff": int(kd),
                "radiant_towers_lost": int(rt),
                "dire_towers_lost": int(dt),
                "radiant_rax_lost": int(rr),
                "dire_rax_lost": int(dr),
                "roshan_radiant": int(rros),
                "roshan_dire": int(dros),
                "radiant_heroes": list(rad_heroes),
                "dire_heroes": list(dire_heroes),
                "radiant_win": label,
            })
        return out

    def vectorize(
        self,
        snapshots: Sequence[Mapping[str, Any]],
        hero_ids: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        hero_index = {int(h): i for i, h in enumerate(hero_ids)}
        n_h = len(hero_ids)
        names = list(NUMERIC_FEATURES) + [f"r_hero_{h}" for h in hero_ids] + [f"d_hero_{h}" for h in hero_ids]
        X = np.zeros((len(snapshots), len(names)), dtype=np.float32)
        y = np.zeros(len(snapshots), dtype=np.int32)
        groups = np.zeros(len(snapshots), dtype=np.int64)
        for i, s in enumerate(snapshots):
            for j, fn in enumerate(NUMERIC_FEATURES):
                X[i, j] = float(s.get(fn, 0) or 0)
            for hid in s.get("radiant_heroes") or []:
                k = hero_index.get(int(hid))
                if k is not None:
                    X[i, len(NUMERIC_FEATURES) + k] = 1.0
            for hid in s.get("dire_heroes") or []:
                k = hero_index.get(int(hid))
                if k is not None:
                    X[i, len(NUMERIC_FEATURES) + n_h + k] = 1.0
            y[i] = int(s.get("radiant_win") or 0)
            groups[i] = int(s.get("match_id") or i)
        return X, y, groups, names

    def encode_query(
        self,
        query: Mapping[str, Any],
        hero_ids: Sequence[int],
        feature_names: Sequence[str],
    ) -> np.ndarray:
        dummy = {
            "match_id": 0,
            "minute": int(query.get("minute") or 0),
            "radiant_gold_adv": float(query.get("radiant_gold_adv") or 0),
            "radiant_xp_adv": float(query.get("radiant_xp_adv") or 0),
            "kills_diff": int(query.get("kills_diff") or 0),
            "radiant_towers_lost": int(query.get("radiant_towers_lost") or 0),
            "dire_towers_lost": int(query.get("dire_towers_lost") or 0),
            "radiant_rax_lost": int(query.get("radiant_rax_lost") or 0),
            "dire_rax_lost": int(query.get("dire_rax_lost") or 0),
            "roshan_radiant": int(query.get("roshan_radiant") or 0),
            "roshan_dire": int(query.get("roshan_dire") or 0),
            "radiant_heroes": [int(x) for x in (query.get("radiant_heroes") or []) if x],
            "dire_heroes": [int(x) for x in (query.get("dire_heroes") or []) if x],
            "radiant_win": 0,
        }
        X, _, _, names = self.vectorize([dummy], hero_ids)
        if list(names) != list(feature_names):
            col = {n: i for i, n in enumerate(names)}
            aligned = np.zeros((1, len(feature_names)), dtype=np.float32)
            for j, n in enumerate(feature_names):
                if n in col:
                    aligned[0, j] = X[0, col[n]]
            return aligned
        return X


# =============================================================================
# 5. 胜率模型
# =============================================================================
class LightGBMWinModel:
    """优先 LightGBM；导入失败时回退 GradientBoostingClassifier。"""

    def __init__(self) -> None:
        self.model: Any = None
        self.backend: str = "lightgbm" if HAS_LIGHTGBM else "sklearn_gbdt"
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
        # 按「整场比赛」分组划分，并在比赛层面按胜负分层，避免训练/测试基础胜率严重偏离
        rng = np.random.RandomState(42)
        match_ids = []
        match_label = {}
        for g, yi in zip(groups.tolist(), y.tolist()):
            g = int(g)
            if g not in match_label:
                match_ids.append(g)
                match_label[g] = int(yi)
        pos = [m for m in match_ids if match_label[m] == 1]
        neg = [m for m in match_ids if match_label[m] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_test_pos = max(1, int(round(len(pos) * 0.2)))
        n_test_neg = max(1, int(round(len(neg) * 0.2)))
        test_matches = set(pos[:n_test_pos] + neg[:n_test_neg])
        test_idx = np.array([i for i, g in enumerate(groups.tolist()) if int(g) in test_matches], dtype=int)
        train_idx = np.array([i for i, g in enumerate(groups.tolist()) if int(g) not in test_matches], dtype=int)
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        n_train_matches = int(len(set(groups[train_idx].tolist())))
        n_test_matches = int(len(set(groups[test_idx].tolist())))

        if HAS_LIGHTGBM:
            self.backend = "lightgbm"
            self.model = lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=24,
                max_depth=5,
                min_child_samples=60,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier

            self.backend = "sklearn_gbdt"
            log.warning("未安装 lightgbm，回退到 sklearn GradientBoostingClassifier")
            self.model = GradientBoostingClassifier(
                n_estimators=180, learning_rate=0.06, max_depth=3, random_state=42
            )

        self.model.fit(Xtr, ytr)
        proba = self._positive_proba(Xte)
        metrics: Dict[str, Any] = {
            "n_snapshots": int(len(y)),
            "n_train_snapshots": int(len(ytr)),
            "n_test_snapshots": int(len(yte)),
            "n_train_matches": n_train_matches,
            "n_test_matches": n_test_matches,
            "backend": self.backend,
        }
        if len(set(yte.tolist())) >= 2:
            metrics["auc"] = float(roc_auc_score(yte, proba))
            metrics["logloss"] = float(log_loss(yte, np.clip(proba, 1e-6, 1 - 1e-6)))
            metrics["brier"] = float(brier_score_loss(yte, proba))
        else:
            metrics["auc"] = None
            metrics["logloss"] = None
            metrics["brier"] = None
            metrics["warning"] = "测试集标签单一，无法计算 AUC"
        metrics["test_win_rate"] = float(np.mean(yte))
        metrics["train_win_rate"] = float(np.mean(ytr))
        return metrics

    def _positive_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型尚未训练或加载")
        return self._positive_proba(X)

    def save(self, path: Path, meta: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "backend": self.backend, "feature_names": self.feature_names}, path)
        path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LightGBMWinModel":
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj.backend = blob.get("backend", "unknown")
        obj.feature_names = blob.get("feature_names") or []
        return obj


# =============================================================================
# 6. 出装推荐（样本频次 / 胜率）
# =============================================================================
@dataclass
class PurchaseRecord:
    hero_id: int
    minute: int
    owned: Tuple[str, ...]
    nxt: str
    won: bool


class FrequencyItemRecommender:
    def __init__(self) -> None:
        self.records: List[PurchaseRecord] = []
        self.stats: Dict[str, Any] = {}
        self.item_meta: Dict[str, Dict[str, Any]] = {}

    def fit(self, matches: Iterable[Mapping[str, Any]], item_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        self.item_meta = item_meta or {}
        recs: List[PurchaseRecord] = []
        n_players = 0
        for m in matches:
            win = bool(m.get("radiant_win"))
            for p in m.get("players") or []:
                hid = p.get("hero_id")
                if not hid:
                    continue
                is_rad = bool(p.get("isRadiant") if p.get("isRadiant") is not None else int(p.get("player_slot", 0)) < 128)
                won = win if is_rad else (not win)
                log_ = p.get("purchase_log") or []
                owned: List[str] = []
                seen_next = False
                for ev in sorted(log_, key=lambda x: int(x.get("time") or 0)):
                    key = (ev.get("key") or "").lower()
                    if is_skipped_item(key):
                        continue
                    t = int(ev.get("time") or 0)
                    minute = max(0, t // 60)
                    recs.append(PurchaseRecord(int(hid), minute, tuple(owned), key, won))
                    if key not in owned:
                        owned.append(key)
                    seen_next = True
                if seen_next:
                    n_players += 1
        self.records = recs
        self.stats = {
            "n_records": len(recs),
            "n_players_with_purchases": n_players,
            "n_unique_heroes": len({r.hero_id for r in recs}),
            "n_unique_items": len({r.nxt for r in recs}),
        }
        log.info("出装样本: %s 条购买记录，覆盖 %s 名英雄", self.stats["n_records"], self.stats["n_unique_heroes"])
        return self.stats

    def recommend(self, hero_id: int, minute: int, current_items: Sequence[str], top_k: int = 8) -> Dict[str, Any]:
        cur = tuple(sorted({c.lower() for c in current_items if c and not is_skipped_item(c)}))
        cur_set = set(cur)
        windows = [6, 12, 25, 99]
        used_window = None
        agg: Dict[str, Dict[str, int]] = {}
        matched = 0
        for w in windows:
            agg = {}
            matched = 0
            for r in self.records:
                if r.hero_id != int(hero_id):
                    continue
                if abs(r.minute - int(minute)) > w:
                    continue
                if r.nxt in cur_set:
                    continue
                owned_set = set(r.owned)
                if cur_set and not cur_set.issubset(owned_set) and len(cur_set & owned_set) < max(1, len(cur_set) - 1):
                    continue
                overlap = len(cur_set & owned_set)
                weight = 1 + overlap
                slot = agg.setdefault(r.nxt, {"count": 0, "wins": 0, "weight": 0, "win_weight": 0})
                slot["count"] += 1
                slot["wins"] += int(r.won)
                slot["weight"] += weight
                slot["win_weight"] += weight * int(r.won)
                matched += 1
            if matched >= 8 or w == windows[-1]:
                used_window = w
                break
        rows = []
        for item, s in agg.items():
            wr = (s["win_weight"] / s["weight"]) if s["weight"] else 0.0
            wr_s = (s["wins"] + 2) / (s["count"] + 4)
            score = s["weight"] * (0.35 + 0.65 * wr_s)
            meta = self.item_meta.get(item) or {}
            rows.append({
                "item": item,
                "dname": meta.get("dname") or item,
                "cost": meta.get("cost"),
                "count": int(s["count"]),
                "wins": int(s["wins"]),
                "winrate": round(wr, 4),
                "smoothed_winrate": round(wr_s, 4),
                "score": round(float(score), 4),
            })
        rows.sort(key=lambda x: (-x["score"], -x["count"]))
        for i, r in enumerate(rows[:top_k], start=1):
            r["rank"] = i
        return {
            "hero_id": int(hero_id),
            "minute": int(minute),
            "current_items": list(cur),
            "matched_records": matched,
            "minute_window": used_window,
            "note": "基于本样本 purchase_log 的频次与胜率，不是序列生成模型；请同时看样本数。",
            "items": rows[:top_k],
        }

    def save(self, path: Path) -> None:
        joblib.dump({"records": self.records, "stats": self.stats, "item_meta": self.item_meta}, path)

    @classmethod
    def load(cls, path: Path) -> "FrequencyItemRecommender":
        blob = joblib.load(path)
        obj = cls()
        obj.records = blob["records"]
        obj.stats = blob.get("stats") or {}
        obj.item_meta = blob.get("item_meta") or {}
        return obj


# =============================================================================
# 7. 目标建议（特征 +1 启发式，非因果）
# =============================================================================
class HeuristicObjectiveAdvisor:
    """把「再拿一座塔 / 再拿一次肉山」写成特征扰动，比较模型胜率变化。"""

    def advise(
        self,
        query: Mapping[str, Any],
        win_model: WinModel,
        builder: FeatureBuilder,
        hero_ids: Sequence[int],
        feature_names: Sequence[str],
    ) -> Dict[str, Any]:
        perspective = (query.get("perspective") or "radiant").lower()
        if perspective not in ("radiant", "dire"):
            perspective = "radiant"

        def _prob(q: Mapping[str, Any]) -> float:
            X = builder.encode_query(q, hero_ids, feature_names)
            p_rad = float(win_model.predict_proba(X)[0])
            return p_rad if perspective == "radiant" else (1.0 - p_rad)

        base = _prob(query)
        q_tower = dict(query)
        if perspective == "radiant":
            q_tower["dire_towers_lost"] = int(query.get("dire_towers_lost") or 0) + 1
        else:
            q_tower["radiant_towers_lost"] = int(query.get("radiant_towers_lost") or 0) + 1
        p_tower = _prob(q_tower)

        q_rosh = dict(query)
        if perspective == "radiant":
            q_rosh["roshan_radiant"] = int(query.get("roshan_radiant") or 0) + 1
        else:
            q_rosh["roshan_dire"] = int(query.get("roshan_dire") or 0) + 1
        p_rosh = _prob(q_rosh)

        d_tower = p_tower - base
        d_rosh = p_rosh - base
        if d_rosh > d_tower + 1e-6:
            preferred = "roshan"
        elif d_tower > d_rosh + 1e-6:
            preferred = "tower"
        else:
            preferred = "uncertain"

        disclaimer = (
            "这是把目标计数特征 +1 后的模型胜率差，属于关联启发式，"
            "不是「现在打肉山/推塔」的因果效应，也没有模拟神符、买活、视野和团队执行。"
        )
        return {
            "perspective": perspective,
            "baseline_win_prob": round(base, 4),
            "take_roshan": {
                "win_prob": round(p_rosh, 4),
                "delta": round(d_rosh, 4),
                "feature_touched": "roshan_radiant" if perspective == "radiant" else "roshan_dire",
            },
            "push_tower": {
                "win_prob": round(p_tower, 4),
                "delta": round(d_tower, 4),
                "feature_touched": "dire_towers_lost" if perspective == "radiant" else "radiant_towers_lost",
            },
            "preferred": preferred,
            "disclaimer": disclaimer,
            "is_causal": False,
        }


# =============================================================================
# 8. 分区 / 刷钱提示（对线热力图 + 录像坐标预留）
# =============================================================================
class HeatmapZonePredictor:
    """用 lane_pos / lane_role 做粗粒度提示，仅反映对线期活动范围。"""

    def __init__(self) -> None:
        self.lane_roles: Dict[int, Counter] = defaultdict(Counter)
        self.zones: Dict[int, Counter] = defaultdict(Counter)
        self.zones_side: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self.stats: Dict[str, Any] = {}

    def fit(self, matches: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        n = 0
        for m in matches:
            for p in m.get("players") or []:
                hid = p.get("hero_id")
                if not hid:
                    continue
                hid = int(hid)
                role = int(p.get("lane_role") or 0)
                self.lane_roles[hid][role] += 1
                is_rad = 1 if (p.get("isRadiant") if p.get("isRadiant") is not None else int(p.get("player_slot", 0)) < 128) else 0
                pos = p.get("lane_pos") or {}
                if isinstance(pos, dict):
                    for xs, ys in pos.items():
                        try:
                            x = float(xs)
                        except Exception:
                            continue
                        if not isinstance(ys, dict):
                            continue
                        for yk, c in ys.items():
                            try:
                                y = float(yk)
                                cnt = int(c)
                            except Exception:
                                continue
                            z = cell_to_zone(x, y)
                            self.zones[hid][z] += cnt
                            self.zones_side[(hid, is_rad)][z] += cnt
                n += 1
        self.stats = {
            "n_players": n,
            "n_heroes": len(self.lane_roles),
            "note": "lane_pos 主要覆盖对线期，不是全场轨迹。",
        }
        log.info("分区统计: %s 名选手, %s 个英雄", n, len(self.lane_roles))
        return self.stats

    def predict(
        self,
        hero_id: int,
        minute: int,
        is_radiant: bool = True,
        lane_role: Optional[int] = None,
    ) -> Dict[str, Any]:
        hid = int(hero_id)
        role_c = self.lane_roles.get(hid, Counter())
        if lane_role is None and role_c:
            lane_role = int(role_c.most_common(1)[0][0])
        role_total = sum(role_c.values()) or 1
        role_dist = [
            {"lane_role": int(k), "name_zh": LANE_ROLE_ZH.get(int(k), "未知"), "count": int(v), "share": round(v / role_total, 3)}
            for k, v in role_c.most_common()
        ]
        zc = self.zones_side.get((hid, 1 if is_radiant else 0)) or self.zones.get(hid) or Counter()
        z_total = sum(zc.values()) or 1
        zone_dist = [
            {"zone": k, "name_zh": ZONE_ZH.get(k, k), "count": int(v), "share": round(v / z_total, 3)}
            for k, v in zc.most_common()
        ]
        if int(minute) <= 12:
            phase = "laning"
            hint = (
                "对线期粗提示：该英雄在样本中最常见分路为"
                f"「{LANE_ROLE_ZH.get(int(lane_role or 0), '未知')}」。"
            )
            if zone_dist:
                hint += f"热力最高区域是{zone_dist[0]['name_zh']}（占比 {zone_dist[0]['share']:.0%}）。"
        else:
            phase = "post_laning"
            hint = (
                "分钟数已超过对线期。OpenDota 的 lane_pos 几乎只记录前期位置，"
                "不能当作当前应去哪边刷钱的实时建议。后期刷钱路线需要录像逐秒坐标，"
                "请通过 ReplayCoordinateZonePredictor 扩展。"
            )
            if zone_dist:
                hint += f" 仅供参考的前期高频区域：{zone_dist[0]['name_zh']}。"
        return {
            "hero_id": hid,
            "minute": int(minute),
            "is_radiant": bool(is_radiant),
            "phase": phase,
            "typical_lane_role": int(lane_role or 0),
            "typical_lane_role_zh": LANE_ROLE_ZH.get(int(lane_role or 0), "未知"),
            "lane_role_distribution": role_dist,
            "zone_distribution": zone_dist,
            "hint": hint,
            "coarse": True,
            "source": "lane_pos/lane_role heatmap",
            "limitation": "对线期热力图粗提示，不是实时坐标，也不是最优刷钱路径。",
        }

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "lane_roles": dict(self.lane_roles),
                "zones": dict(self.zones),
                "zones_side": dict(self.zones_side),
                "stats": self.stats,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "HeatmapZonePredictor":
        blob = joblib.load(path)
        obj = cls()
        obj.lane_roles = defaultdict(Counter, {int(k): Counter(v) for k, v in blob.get("lane_roles", {}).items()})
        obj.zones = defaultdict(Counter, {int(k): Counter(v) for k, v in blob.get("zones", {}).items()})
        raw_side = blob.get("zones_side", {})
        fixed: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        for k, v in raw_side.items():
            if isinstance(k, (list, tuple)) and len(k) == 2:
                fixed[(int(k[0]), int(k[1]))] = Counter(v)
        obj.zones_side = defaultdict(Counter, fixed)
        obj.stats = blob.get("stats") or {}
        return obj


class ReplayCoordinateZonePredictor(HeatmapZonePredictor):
    """
    预留接口：未来用录像解析库解析 .dem 后，按每秒坐标预测刷钱区域。

    建议实现步骤（尚未编码，请勿在此假装已完成）：
      1. 从比赛 JSON 的 replay_url 下载录像；
      2. 解析 tick 到 (hero_id, x, y, gold)；
      3. 按分钟/野区栅格统计效率，训练时空模型；
      4. 覆盖 predict()，并在返回中把 source 改为 replay_coordinates。
    """

    def predict(self, hero_id: int, minute: int, is_radiant: bool = True, lane_role: Optional[int] = None) -> Dict[str, Any]:
        base = super().predict(hero_id, minute, is_radiant, lane_role)
        base["replay_interface"] = {
            "implemented": False,
            "hook": "继承 ReplayCoordinateZonePredictor 并覆盖 predict()，在 load_context() 中替换实例。",
            "expected_input": "逐秒坐标轨迹（来自录像文件），而非 lane_pos。",
        }
        return base


# =============================================================================
# 9. 文本解释器（模板桩 + 环境变量钩子，绝不硬编码密钥）
# =============================================================================
class TemplateLLMExplainer:
    def explain(self, payload: Mapping[str, Any]) -> str:
        win = payload.get("win") or {}
        obj = payload.get("objective") or {}
        items = payload.get("items") or {}
        zone = payload.get("zone") or {}
        p = win.get("perspective_win_prob")
        pref = obj.get("preferred")
        recs = (items.get("items") or [])[:3]
        rec_txt = "、".join(f"{r.get('dname')}(n={r.get('count')})" for r in recs) or "暂无足够样本"
        pref_zh = {"roshan": "打肉山的特征扰动更高", "tower": "推塔的特征扰动更高", "uncertain": "两者差别很小"}.get(pref, "难以判断")
        p_txt = f"{p:.1%}" if isinstance(p, (int, float)) else "未知"
        zhint = zone.get("hint") or ""
        return (
            f"当前视角估计胜率约 {p_txt}。"
            f"出装方面，样本中较常见的下一步是 {rec_txt}。"
            f"目标选择上，{pref_zh}（仅启发式，非因果）。"
            f"{zhint}"
        )


class OptionalHttpLLMExplainer:
    """
    若设置了 DOTA_ASSIST_LLM_BASE_URL 与 DOTA_ASSIST_LLM_API_KEY，则尝试调用
    兼容 OpenAI 的 chat completions；失败时自动回退到模板句。
    密钥只从环境变量读取，不会写入磁盘或日志。
    """

    def __init__(self) -> None:
        self.fallback = TemplateLLMExplainer()
        self.base = os.environ.get("DOTA_ASSIST_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("DOTA_ASSIST_LLM_API_KEY", "")
        self.model = os.environ.get("DOTA_ASSIST_LLM_MODEL", "gpt-4o-mini")
        self.path = os.environ.get("DOTA_ASSIST_LLM_PATH", "/chat/completions")

    def explain(self, payload: Mapping[str, Any]) -> str:
        if not (self.base and self.key):
            return self.fallback.explain(payload)
        url = self.base + (self.path if self.path.startswith("/") else "/" + self.path)
        prompt = (
            "你是 Dota 2 复盘助手。用不超过 120 字的中文，把下列模型输出翻译成谨慎建议，"
            "必须提醒这不是对局中的实时辅助，且目标比较不是因果结论。\n"
            + json.dumps(payload, ensure_ascii=False)[:4000]
        )
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a cautious Dota 2 coaching assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            txt = data["choices"][0]["message"]["content"].strip()
            return txt or self.fallback.explain(payload)
        except Exception as e:
            log.warning("外部语言模型调用失败，回退模板: %s", type(e).__name__)
            return self.fallback.explain(payload)


# =============================================================================
# 10. 本地游戏状态映射（结构预留，仅用于训练/回放研究）
# =============================================================================
# 示例配置文本（放到游戏 cfg 目录后，客户端会向本机端口推送 JSON）。
# 警告：仅建议在训练模式、回放或自定义中使用，不要接入天梯或正式比赛。
#
# filename: gamestate_integration_dota_assist.cfg
# "dota_assist" {
#     "uri" "http://127.0.0.1:8000/gsi"
#     "timeout" "5.0"
#     "buffer" "0.1"
#     "throttle" "1.0"
#     "heartbeat" "30.0"
#     "data" { "provider" "1" "map" "1" "player" "1" "hero" "1"
#              "abilities" "1" "items" "1" "buildings" "1" "draft" "1" }
# }
GSI_CFG_EXAMPLE = (
    "See comment block above in section 10. Intended for practice/replay only."
)


def _extract_item_slots(items_block: Mapping[str, Any]) -> List[str]:
    names = []
    if not isinstance(items_block, dict):
        return names
    for slot, val in items_block.items():
        if not isinstance(val, dict):
            continue
        if not str(slot).startswith("slot") and not str(slot).startswith("inventory"):
            continue
        name = (val.get("name") or "").replace("item_", "")
        if name and name not in ("empty", "item_empty") and not is_skipped_item(name):
            names.append(name)
    return names


def map_gsi_to_features(gsi: Mapping[str, Any], hero_lookup: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
    """
    把本地推送的对局状态 JSON 尽量映射到本原型的局面特征。
    玩家视角下大量敌方信息不可见，字段会缺失，不能假装拥有全局视角。
    """
    mp = gsi.get("map") or {}
    clock = mp.get("clock_time")
    if clock is None:
        clock = mp.get("game_time") or 0
    try:
        minute = max(0, int(clock) // 60)
    except Exception:
        minute = 0

    rad, dire = [], []
    allplayers = gsi.get("allplayers") or gsi.get("player") or {}
    if isinstance(allplayers, dict):
        if "heroname" in allplayers or "hero" in allplayers or ("name" in allplayers and "team" in allplayers):
            candidates = [allplayers]
        else:
            candidates = [v for v in allplayers.values() if isinstance(v, dict)]
        for p in candidates:
            team = str(p.get("team") or p.get("team_name") or "").lower()
            hid = p.get("hero_id")
            hero_obj = p.get("hero") if isinstance(p.get("hero"), dict) else {}
            hid = hid or hero_obj.get("id") or p.get("id")
            hname = p.get("heroname") or p.get("hero_name") or hero_obj.get("name")
            if not hid and hname and hero_lookup:
                hid = hero_lookup.get(str(hname).replace("npc_dota_hero_", "")) or hero_lookup.get(str(hname))
            if not hid:
                continue
            if "radiant" in team or team in ("0", "good", "goodguys"):
                rad.append(int(hid))
            elif "dire" in team or team in ("1", "bad", "badguys"):
                dire.append(int(hid))

    buildings = gsi.get("buildings") or {}

    def _lost(side: str, kind: str) -> int:
        block = buildings.get(side) or {}
        lost = 0
        if not isinstance(block, dict):
            return 0
        for name, st in block.items():
            if kind == "tower" and "tower" not in name.lower():
                continue
            if kind == "rax" and "rax" not in name.lower() and "barracks" not in name.lower():
                continue
            if isinstance(st, dict) and (st.get("health") == 0 or st.get("alive") is False):
                lost += 1
        return lost

    hero = gsi.get("hero") or {}
    items = _extract_item_slots(gsi.get("items") or {})
    return {
        "source": "local_game_state",
        "visible_scope": "limited_to_client_perspective",
        "minute": minute,
        "clock_time": clock,
        "game_state": mp.get("game_state") or mp.get("gamestate"),
        "radiant_heroes": rad,
        "dire_heroes": dire,
        "radiant_towers_lost": _lost("radiant", "tower"),
        "dire_towers_lost": _lost("dire", "tower"),
        "radiant_rax_lost": _lost("radiant", "rax"),
        "dire_rax_lost": _lost("dire", "rax"),
        "radiant_gold_adv": None,
        "radiant_xp_adv": None,
        "kills_diff": None,
        "own_hero_id": hero.get("id"),
        "current_items": items,
        "note": "玩家视角通常没有完整的经济差、敌方阵容与坐标，映射结果可能不完整。",
    }


# =============================================================================
# 11. 训练流程与应用装配
# =============================================================================
@dataclass
class AppContext:
    data: OpenDotaDataSource
    builder: SnapshotFeatureBuilder
    win_model: LightGBMWinModel
    item_rec: FrequencyItemRecommender
    advisor: HeuristicObjectiveAdvisor
    zone: ZonePredictor
    explainer: LLMExplainer
    heroes: List[Dict[str, Any]] = field(default_factory=list)
    items: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hero_ids: List[int] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)


def train_pipeline(data: Optional[OpenDotaDataSource] = None) -> Dict[str, Any]:
    ensure_dirs()
    data = data or OpenDotaDataSource()
    data.fetch_constants()
    heroes = data.load_heroes()
    items = data.load_items()
    hero_ids = [h["id"] for h in heroes]
    builder = SnapshotFeatureBuilder()

    matches = list(data.iter_cached_matches())
    if len(matches) < 30:
        raise RuntimeError(f"可用比赛不足（{len(matches)}），请先运行: python dota_assist.py fetch")

    snapshots: List[Dict[str, Any]] = []
    for m in matches:
        snapshots.extend(builder.build_snapshots(m))
    if len(snapshots) < 200:
        raise RuntimeError(f"快照过少（{len(snapshots)}），请检查缓存比赛是否已被解析。")

    X_all, y, groups, names_all = builder.vectorize(snapshots, hero_ids)
    # 胜率模型默认只用数值特征。完整英雄 one-hot 仍由 FeatureBuilder 生成，
    # 但在约 200 场分组划分下会严重过拟合，因此不进入默认模型。
    used_names = list(NUMERIC_FEATURES)
    col = {n: i for i, n in enumerate(names_all)}
    X = np.column_stack([X_all[:, col[n]] for n in used_names])
    win = LightGBMWinModel()
    win.feature_names = used_names
    metrics = win.fit(X, y, groups)
    metrics.update({
        "n_matches": len(matches),
        "n_heroes": len(hero_ids),
        "n_features": len(used_names),
        "available_features": len(names_all),
        "hero_encoding": "one-hot constructed but excluded from default win model due to sparsity",
        "version": APP_VERSION,
        "grouped_split": True,
        "split_unit": "match_id",
        "compliance": "仅用于复盘/训练/教练，不要作为正式比赛中的实时辅助。",
    })
    log.info(
        "胜率模型 %s  | 场次=%s 快照=%s 测试AUC=%s logloss=%s brier=%s",
        metrics["backend"], metrics["n_matches"], metrics["n_snapshots"],
        metrics.get("auc"), metrics.get("logloss"), metrics.get("brier"),
    )

    item_rec = FrequencyItemRecommender()
    item_stats = item_rec.fit(matches, items)
    zone = ReplayCoordinateZonePredictor()
    zone_stats = zone.fit(matches)

    rng = random.Random(42)
    candidates = [s for s in snapshots if 8 <= int(s["minute"]) <= 40]
    sample_snaps = rng.sample(candidates, k=min(24, len(candidates)))
    slim_samples = []
    keys = [
        "match_id", "minute", "radiant_gold_adv", "radiant_xp_adv", "kills_diff",
        "radiant_towers_lost", "dire_towers_lost", "radiant_rax_lost", "dire_rax_lost",
        "roshan_radiant", "roshan_dire", "radiant_heroes", "dire_heroes", "radiant_win",
    ]
    for s in sample_snaps:
        slim_samples.append({k: s[k] for k in keys})

    (ART_DIR / "heroes.json").write_text(json.dumps(heroes, ensure_ascii=False, indent=2), encoding="utf-8")
    item_list = list(items.values())
    item_list.sort(key=lambda x: (x.get("skipped") is True, x.get("cost") is None, x.get("cost") or 0, x.get("dname") or ""))
    (ART_DIR / "items.json").write_text(json.dumps(item_list, ensure_ascii=False, indent=2), encoding="utf-8")
    (ART_DIR / "samples.json").write_text(json.dumps(slim_samples, ensure_ascii=False), encoding="utf-8")
    (ART_DIR / "feature_names.json").write_text(json.dumps(used_names, ensure_ascii=False), encoding="utf-8")

    meta = dict(metrics)
    meta["item_stats"] = item_stats
    meta["zone_stats"] = zone_stats
    meta["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    win.save(ART_DIR / "win_model.joblib", meta)
    item_rec.save(ART_DIR / "item_recommender.joblib")
    zone.save(ART_DIR / "zone_predictor.joblib")
    (ART_DIR / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("训练产物已写入 %s", ART_DIR)
    return meta


def load_context() -> AppContext:
    ensure_dirs()
    data = OpenDotaDataSource()
    heroes_path = ART_DIR / "heroes.json"
    items_path = ART_DIR / "items.json"
    if not (ART_DIR / "win_model.joblib").exists():
        raise RuntimeError("未找到训练产物，请先运行: python dota_assist.py train")
    if heroes_path.exists():
        heroes = json.loads(heroes_path.read_text(encoding="utf-8"))
    else:
        heroes = data.load_heroes()
    if items_path.exists():
        item_list = json.loads(items_path.read_text(encoding="utf-8"))
        items = {it["key"]: it for it in item_list}
    else:
        items = data.load_items()
    win = LightGBMWinModel.load(ART_DIR / "win_model.joblib")
    if not win.feature_names and (ART_DIR / "feature_names.json").exists():
        win.feature_names = json.loads((ART_DIR / "feature_names.json").read_text(encoding="utf-8"))
    item_rec = FrequencyItemRecommender.load(ART_DIR / "item_recommender.joblib")
    item_rec.item_meta = items
    zone = ReplayCoordinateZonePredictor.load(ART_DIR / "zone_predictor.joblib")
    meta = {}
    if (ART_DIR / "model_meta.json").exists():
        meta = json.loads((ART_DIR / "model_meta.json").read_text(encoding="utf-8"))
    samples = []
    if (ART_DIR / "samples.json").exists():
        samples = json.loads((ART_DIR / "samples.json").read_text(encoding="utf-8"))
    return AppContext(
        data=data,
        builder=SnapshotFeatureBuilder(),
        win_model=win,
        item_rec=item_rec,
        advisor=HeuristicObjectiveAdvisor(),
        zone=zone,
        explainer=OptionalHttpLLMExplainer(),
        heroes=heroes,
        items=items,
        hero_ids=[h["id"] for h in heroes],
        feature_names=list(win.feature_names),
        meta=meta,
        samples=samples,
    )


# =============================================================================
# 12. FastAPI 服务
# =============================================================================
def create_app(ctx: Optional[AppContext] = None) -> FastAPI:
    app = FastAPI(title="Dota Assist Prototype", version=APP_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state: Dict[str, Any] = {"ctx": ctx}

    def get_ctx() -> AppContext:
        if state["ctx"] is None:
            state["ctx"] = load_context()
        return state["ctx"]

    @app.get("/")
    def index():
        if not INDEX_HTML.exists():
            return JSONResponse({"error": "找不到 index.html"}, status_code=500)
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")

    @app.get("/api/heroes")
    def api_heroes():
        return {"heroes": get_ctx().heroes}

    @app.get("/api/items")
    def api_items(include_skipped: bool = False):
        items = list(get_ctx().items.values())
        if not include_skipped:
            items = [it for it in items if not it.get("skipped")]
        items.sort(key=lambda x: (x.get("cost") is None, x.get("cost") or 0, x.get("dname") or ""))
        return {"items": items}

    @app.get("/api/model_info")
    def api_model_info():
        ctx = get_ctx()
        m = dict(ctx.meta)
        m["version"] = APP_VERSION
        m["feature_count"] = len(ctx.feature_names)
        m["compliance"] = (
            "Valve 限制在比赛中提供额外决策优势的第三方工具。"
            "本原型用于复盘、训练与教练分析，不要接入天梯或正式比赛。"
        )
        return m

    @app.get("/api/sample_snapshot")
    def api_sample_snapshot():
        ctx = get_ctx()
        if not ctx.samples:
            return JSONResponse({"error": "没有可用的真实快照"}, status_code=404)
        s = random.choice(ctx.samples)
        return {"snapshot": s, "note": "来自训练样本的真实对局分钟切片，不是编造数据。"}

    def _require_lineup(body: Mapping[str, Any]) -> Optional[str]:
        rad = body.get("radiant_heroes") or []
        dire = body.get("dire_heroes") or []
        if len([x for x in rad if x]) != 5 or len([x for x in dire if x]) != 5:
            return "请提供天辉和夜魇各 5 个英雄 ID"
        return None

    @app.post("/api/predict_win")
    def api_predict_win(body: Dict[str, Any]):
        err = _require_lineup(body)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        ctx = get_ctx()
        X = ctx.builder.encode_query(body, ctx.hero_ids, ctx.feature_names)
        p_rad = float(ctx.win_model.predict_proba(X)[0])
        perspective = (body.get("perspective") or "radiant").lower()
        p_view = p_rad if perspective == "radiant" else (1 - p_rad)
        return {
            "radiant_win_prob": round(p_rad, 4),
            "dire_win_prob": round(1 - p_rad, 4),
            "perspective": perspective,
            "perspective_win_prob": round(p_view, 4),
            "features_used": NUMERIC_FEATURES,
        }

    @app.post("/api/recommend_items")
    def api_recommend_items(body: Dict[str, Any]):
        hero_id = body.get("hero_id")
        if not hero_id:
            return JSONResponse({"error": "需要 hero_id"}, status_code=400)
        minute = int(body.get("minute") or 0)
        current = body.get("current_items") or []
        ctx = get_ctx()
        return ctx.item_rec.recommend(int(hero_id), minute, current, top_k=int(body.get("top_k") or 8))

    @app.post("/api/objective_advice")
    def api_objective_advice(body: Dict[str, Any]):
        err = _require_lineup(body)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        ctx = get_ctx()
        return ctx.advisor.advise(body, ctx.win_model, ctx.builder, ctx.hero_ids, ctx.feature_names)

    @app.post("/api/zone_hint")
    def api_zone_hint(body: Dict[str, Any]):
        hero_id = body.get("hero_id")
        if not hero_id:
            return JSONResponse({"error": "需要 hero_id"}, status_code=400)
        ctx = get_ctx()
        return ctx.zone.predict(
            int(hero_id),
            int(body.get("minute") or 0),
            is_radiant=bool(body.get("is_radiant", True)),
            lane_role=body.get("lane_role"),
        )

    @app.post("/api/explain")
    def api_explain(body: Dict[str, Any]):
        ctx = get_ctx()
        return {"text": ctx.explainer.explain(body), "backend": type(ctx.explainer).__name__}

    @app.post("/gsi")
    async def local_state_receiver(request: Request):
        """接收本地推送的对局状态 JSON。默认只做映射，不把建议写回游戏。"""
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        ctx = get_ctx()
        lookup = {}
        for h in ctx.heroes:
            if h.get("name"):
                lookup[h["name"]] = h["id"]
                lookup[h["name"].replace("npc_dota_hero_", "")] = h["id"]
        mapped = map_gsi_to_features(payload, lookup)
        return {
            "ok": True,
            "mapped": mapped,
            "warning": "不要在天梯或正式比赛中启用本接收器。",
            "cfg_hint": "见源码第 10 节注释中的示例配置。",
        }

    @app.get("/api/health")
    def api_health():
        ok = (ART_DIR / "win_model.joblib").exists()
        return {"ok": ok, "version": APP_VERSION}

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    ctx = load_context()
    app = create_app(ctx)
    log.info("服务启动 http://%s:%s  （按 Ctrl+C 结束）", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# =============================================================================
# 13. 命令行入口
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dota 2 决策辅助原型")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fetch", help="从 OpenDota 拉取并缓存比赛")
    pf.add_argument("--target", type=int, default=DEFAULT_TARGET, help="可用已解析比赛的目标场次")
    pf.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="两次请求间隔秒数")
    pf.add_argument("--item-popularity", action="store_true", help="额外按英雄拉取 itemPopularity")
    sub.add_parser("train", help="基于缓存训练模型并写出 artifacts/")
    ps = sub.add_parser("serve", help="启动 FastAPI 与前端页面")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    pa = sub.add_parser("all", help="fetch + train + serve")
    pa.add_argument("--target", type=int, default=DEFAULT_TARGET)
    pa.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    pa.add_argument("--host", default="127.0.0.1")
    pa.add_argument("--port", type=int, default=8000)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "fetch":
            ds = OpenDotaDataSource(sleep_s=args.sleep)
            ds.fetch_matches(target=args.target, sleep_s=args.sleep)
            if args.item_popularity:
                heroes = ds.load_heroes()
                ds.fetch_item_popularity([h["id"] for h in heroes])
            print(f"可用比赛: {ds.count_usable_cached()}")
            return 0
        if args.cmd == "train":
            meta = train_pipeline()
            public = {k: meta[k] for k in meta if k not in ("item_stats", "zone_stats")}
            print(json.dumps(public, ensure_ascii=False, indent=2))
            return 0
        if args.cmd == "serve":
            serve(host=args.host, port=args.port)
            return 0
        if args.cmd == "all":
            ds = OpenDotaDataSource(sleep_s=args.sleep)
            ds.fetch_matches(target=args.target, sleep_s=args.sleep)
            train_pipeline(ds)
            serve(host=args.host, port=args.port)
            return 0
        return 1
    except Exception as e:
        log.error("%s", e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
