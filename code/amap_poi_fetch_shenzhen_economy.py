#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高德 POI 经济型抓取与 H3 聚合脚本

目标：
1. 尽量减少高德 API 调用次数；
2. 抓取一次后，直接产出后续模型需要的 H3 网格 POI 特征；
3. 默认保存“去重后的最小 POI 表”和“网格聚合表”，不保存大量重复原始响应；
4. 支持估算模式、预算上限、断点续抓和失败重试。

推荐测试：
python amap_poi_fetch_shenzhen_economy.py --amap-key 你的Key --out-dir ./amap_poi_economy_test --grid-summary ./bike_output_bj_corrected/stage2_research_ready/grid_summary_selected.csv --district-filter 南山区 --estimate-only

正式运行：
python amap_poi_fetch_shenzhen_economy.py --amap-key 你的Key --out-dir ./amap_poi_economy --grid-summary ./bike_output_bj_corrected/stage2_research_ready/grid_summary_selected.csv --tile-size 0.02 --max-calls 20000

输出：
- poi_unique_minimal.csv: 去重后的最小 POI 点表，可复核、可重聚合
- poi_grid_features.csv: H3 网格 POI 聚合特征，直接并入模型
- poi_category_summary.csv: POI 类别汇总
- poi_fetch_log.csv: 每个 tile/page 的调用日志
- poi_manifest.json: 本次运行参数和结果清单

说明：
高德地点搜索返回 GCJ-02 坐标。本脚本会转换为 WGS84 后匹配 H3，以便与订单网格一致。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import h3
except ImportError as exc:
    raise ImportError("请先安装 h3：pip install h3") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ===============================
# 坐标转换：GCJ-02 / BD-09 -> WGS84
# ===============================
X_PI = math.pi * 3000.0 / 180.0
PI = math.pi
A = 6378245.0
EE = 0.00669342162296594323


def _out_of_china(lng: float, lat: float) -> bool:
    return lng < 72.004 or lng > 137.8347 or lat < 0.8293 or lat > 55.8271


def _transform_lat(lng: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 * math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng: float, lat: float) -> float:
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 * math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * PI) + 40.0 * math.sin(lng / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * PI) + 300.0 * math.sin(lng / 30.0 * PI)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84_one(lng: float, lat: float) -> Tuple[float, float]:
    if _out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrt_magic) * PI)
    dlng = (dlng * 180.0) / (A / sqrt_magic * math.cos(radlat) * PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat


def bd09_to_gcj02_one(lng: float, lat: float) -> Tuple[float, float]:
    x = lng - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * X_PI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * X_PI)
    return z * math.cos(theta), z * math.sin(theta)


def to_wgs84(lng: float, lat: float, coord_system: str) -> Tuple[float, float]:
    coord_system = coord_system.lower()
    if coord_system in {"wgs84", "wgs", "raw"}:
        return lng, lat
    if coord_system == "gcj02":
        return gcj02_to_wgs84_one(lng, lat)
    if coord_system == "bd09":
        glng, glat = bd09_to_gcj02_one(lng, lat)
        return gcj02_to_wgs84_one(glng, glat)
    raise ValueError("coord_system 只能是 gcj02 / bd09 / wgs84 / raw")


def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    return h3.geo_to_h3(lat, lng, resolution)


# ===============================
# 类型分类
# ===============================
# 为降低调用次数，默认一次查询多个大类 typecode，再根据返回的 type/typecode 分类。
# 注意：高德 typecode 体系很细，下面是论文机制变量的实用归并口径。
DEFAULT_TYPE_CODES = [
    "050000",  # 餐饮服务
    "060000",  # 购物服务
    "070000",  # 生活服务
    "080000",  # 体育休闲服务
    "090000",  # 医疗保健服务
    "110000",  # 风景名胜
    "120000",  # 商务住宅
    "130000",  # 政府机构及社会团体
    "140000",  # 科教文化服务
    "150000",  # 交通设施服务
    "170000",  # 公司企业
]

CATEGORY_ORDER = [
    "residential",
    "office",
    "commercial",
    "metro",
    "bus",
    "education",
    "recreation",
    "public_service",
    "medical",
    "transport_other",
    "other",
]


def classify_poi(type_text: str, typecode: str, name: str = "") -> str:
    text = f"{type_text};{typecode};{name}"
    # 交通先细分
    if re.search(r"地铁|轨道交通|轻轨", text):
        return "metro"
    if re.search(r"公交|巴士|公共汽车", text):
        return "bus"
    if typecode.startswith("150"):
        return "transport_other"
    # 居住/办公
    if re.search(r"住宅区|小区|宿舍|公寓|别墅|社区", text):
        return "residential"
    if re.search(r"写字楼|商务写字楼|产业园|科技园|工业园|公司|企业|园区|大厦", text) or typecode.startswith("17"):
        return "office"
    # 商业消费
    if typecode.startswith(("05", "06", "07")):
        return "commercial"
    # 教育
    if typecode.startswith("14") or re.search(r"学校|大学|学院|幼儿园|培训", text):
        return "education"
    # 休闲
    if typecode.startswith(("08", "11")) or re.search(r"公园|景区|体育|健身|休闲|文化宫|博物馆", text):
        return "recreation"
    # 医疗与公服
    if typecode.startswith("09") or re.search(r"医院|诊所|卫生院|药房", text):
        return "medical"
    if typecode.startswith("13") or re.search(r"政府|派出所|法院|街道办|社区服务|公共服务", text):
        return "public_service"
    if typecode.startswith("12"):
        return "residential"
    return "other"


# ===============================
# Tile 生成与 API
# ===============================
@dataclass
class Config:
    amap_key: str
    out_dir: str
    grid_summary: Optional[str] = None
    tile_size: float = 0.02
    bbox: Optional[str] = None  # min_lng,min_lat,max_lng,max_lat
    district_filter: Optional[str] = None
    offset: int = 25
    max_pages: int = 100
    sleep_seconds: float = 0.05
    max_retry: int = 3
    max_calls: int = 0  # 0 means unlimited
    saturation_threshold: int = 850
    min_tile_size: float = 0.003
    recursive_split: bool = True
    coord_system: str = "gcj02"
    h3_resolution: int = 8
    estimate_only: bool = False
    estimate_sample_tiles: int = 40
    save_unique_poi: bool = True
    resume: bool = True


def parse_bbox(bbox: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox 格式应为 min_lng,min_lat,max_lng,max_lat")
    return parts[0], parts[1], parts[2], parts[3]


def bbox_from_grid_summary(path: str, margin: float = 0.01) -> Tuple[float, float, float, float]:
    df = pd.read_csv(path)
    lng_col = "center_lng" if "center_lng" in df.columns else "grid_center_lng"
    lat_col = "center_lat" if "center_lat" in df.columns else "grid_center_lat"
    if lng_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"grid_summary 缺少中心点字段。现有字段：{list(df.columns)}")
    df = df.dropna(subset=[lng_col, lat_col])
    return (
        float(df[lng_col].min() - margin),
        float(df[lat_col].min() - margin),
        float(df[lng_col].max() + margin),
        float(df[lat_col].max() + margin),
    )


def generate_tiles(bbox: Tuple[float, float, float, float], tile_size: float) -> List[Tuple[float, float, float, float]]:
    min_lng, min_lat, max_lng, max_lat = bbox
    tiles = []
    lng = min_lng
    while lng < max_lng:
        lng2 = min(lng + tile_size, max_lng)
        lat = min_lat
        while lat < max_lat:
            lat2 = min(lat + tile_size, max_lat)
            tiles.append((lng, lat, lng2, lat2))
            lat += tile_size
        lng += tile_size
    return tiles


def tile_to_polygon(tile: Tuple[float, float, float, float]) -> str:
    min_lng, min_lat, max_lng, max_lat = tile
    # v3 支持 lng,lat|lng,lat|...
    return f"{min_lng:.6f},{min_lat:.6f}|{max_lng:.6f},{min_lat:.6f}|{max_lng:.6f},{max_lat:.6f}|{min_lng:.6f},{max_lat:.6f}"


def split_tile(tile: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
    min_lng, min_lat, max_lng, max_lat = tile
    mid_lng = (min_lng + max_lng) / 2
    mid_lat = (min_lat + max_lat) / 2
    return [
        (min_lng, min_lat, mid_lng, mid_lat),
        (mid_lng, min_lat, max_lng, mid_lat),
        (min_lng, mid_lat, mid_lng, max_lat),
        (mid_lng, mid_lat, max_lng, max_lat),
    ]


def request_polygon(session: requests.Session, cfg: Config, tile: Tuple[float, float, float, float], page: int, types: str) -> Dict:
    params = {
        "key": cfg.amap_key,
        "polygon": tile_to_polygon(tile),
        "types": types,
        "offset": cfg.offset,
        "page": page,
        "extensions": "base",
        "output": "json",
    }
    url = "https://restapi.amap.com/v3/place/polygon"
    last_error = None
    for _ in range(cfg.max_retry):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data
        except Exception as exc:  # pragma: no cover
            last_error = exc
            time.sleep(max(cfg.sleep_seconds, 0.2))
    return {"status": "0", "info": str(last_error), "infocode": "PY_ERROR", "pois": [], "count": "0"}


def parse_location(loc: str) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(loc, str) or "," not in loc:
        return None, None
    try:
        lng, lat = loc.split(",")[:2]
        return float(lng), float(lat)
    except Exception:
        return None, None


def safe_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


# ===============================
# 主处理
# ===============================
def load_existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    try:
        for chunk in pd.read_csv(path, usecols=["poi_id"], chunksize=200000):
            ids.update(chunk["poi_id"].astype(str).dropna().tolist())
    except Exception:
        return set()
    return ids


def load_done_tiles(log_path: Path) -> set:
    if not log_path.exists():
        return set()
    try:
        log = pd.read_csv(log_path)
    except Exception:
        return set()
    if "tile_id" not in log.columns or "task_status" not in log.columns:
        return set()
    done = log.loc[log["task_status"].eq("done"), "tile_id"].astype(str).tolist()
    return set(done)


def make_tile_id(tile: Tuple[float, float, float, float]) -> str:
    return "_".join(f"{x:.6f}" for x in tile)


def append_rows_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def estimate_calls(session: requests.Session, cfg: Config, tiles: List[Tuple[float, float, float, float]], types: str, out_dir: Path) -> None:
    n = min(cfg.estimate_sample_tiles, len(tiles))
    sample = random.sample(tiles, n) if len(tiles) > n else tiles
    rows = []
    total_count = 0
    saturated = 0
    calls_used = 0
    for tile in tqdm(sample, desc="估算样本 tile") if tqdm else sample:
        data = request_polygon(session, cfg, tile, 1, types)
        calls_used += 1
        count = safe_int(data.get("count"), 0)
        rows.append({"tile_id": make_tile_id(tile), "count_first_query": count, "status": data.get("status"), "info": data.get("info")})
        total_count += count
        if count >= cfg.saturation_threshold:
            saturated += 1
        time.sleep(cfg.sleep_seconds)
    avg_count = total_count / max(n, 1)
    avg_pages = math.ceil(avg_count / cfg.offset) if cfg.offset else 1
    est_calls_no_split = len(tiles) * max(avg_pages, 1)
    est_poi_raw = avg_count * len(tiles)
    summary = {
        "mode": "estimate_only",
        "sample_tiles": n,
        "total_tiles": len(tiles),
        "api_calls_used_for_estimate": calls_used,
        "avg_count_per_tile_first_query": avg_count,
        "sample_saturated_tiles": saturated,
        "sample_saturation_rate": saturated / max(n, 1),
        "estimated_raw_poi_before_dedup": est_poi_raw,
        "estimated_api_calls_without_recursive_split": est_calls_no_split,
        "note": "若高密度 tile 被递归切分，实际调用可能高于该估算；若去重明显，最终唯一 POI 会低于 raw 估算。",
    }
    pd.DataFrame(rows).to_csv(out_dir / "poi_estimate_sample.csv", index=False, encoding="utf-8-sig")
    (out_dir / "poi_estimate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def process_tile(
    session: requests.Session,
    cfg: Config,
    tile: Tuple[float, float, float, float],
    types: str,
    seen_ids: set,
    unique_path: Path,
    log_path: Path,
    call_counter: Dict[str, int],
    depth: int = 0,
) -> Counter:
    tile_id = make_tile_id(tile)
    min_lng, min_lat, max_lng, max_lat = tile
    tile_width = max(max_lng - min_lng, max_lat - min_lat)
    counts = Counter()

    # first page: count + initial data
    data = request_polygon(session, cfg, tile, 1, types)
    call_counter["calls"] += 1
    status = data.get("status")
    info = data.get("info")
    count = safe_int(data.get("count"), 0)
    pois = data.get("pois") or []

    if status != "1":
        append_rows_csv(log_path, [{
            "tile_id": tile_id, "depth": depth, "page": 1, "count": count,
            "returned": 0, "status": status, "info": info, "task_status": "error",
        }], ["tile_id", "depth", "page", "count", "returned", "status", "info", "task_status"])
        return counts

    # 如果过饱和且 tile 还能切小，则不分页，直接四分，避免结果上限造成漏数
    if cfg.recursive_split and count >= cfg.saturation_threshold and tile_width > cfg.min_tile_size:
        append_rows_csv(log_path, [{
            "tile_id": tile_id, "depth": depth, "page": 1, "count": count,
            "returned": len(pois), "status": status, "info": info, "task_status": "split",
        }], ["tile_id", "depth", "page", "count", "returned", "status", "info", "task_status"])
        for sub in split_tile(tile):
            if cfg.max_calls and call_counter["calls"] >= cfg.max_calls:
                break
            counts.update(process_tile(session, cfg, sub, types, seen_ids, unique_path, log_path, call_counter, depth + 1))
        return counts

    pages = min(cfg.max_pages, max(1, math.ceil(count / cfg.offset)))
    all_pages = [(1, pois)]
    for page in range(2, pages + 1):
        if cfg.max_calls and call_counter["calls"] >= cfg.max_calls:
            break
        d = request_polygon(session, cfg, tile, page, types)
        call_counter["calls"] += 1
        if d.get("status") != "1":
            append_rows_csv(log_path, [{
                "tile_id": tile_id, "depth": depth, "page": page, "count": count,
                "returned": 0, "status": d.get("status"), "info": d.get("info"), "task_status": "page_error",
            }], ["tile_id", "depth", "page", "count", "returned", "status", "info", "task_status"])
            break
        all_pages.append((page, d.get("pois") or []))
        time.sleep(cfg.sleep_seconds)

    unique_rows = []
    for page, page_pois in all_pages:
        for p in page_pois:
            poi_id = str(p.get("id") or "").strip()
            if not poi_id:
                # 没有 ID 时用 name+location 兜底
                poi_id = f"NOID::{p.get('name','')}::{p.get('location','')}"
            if poi_id in seen_ids:
                continue
            lng_gcj, lat_gcj = parse_location(p.get("location", ""))
            if lng_gcj is None or lat_gcj is None:
                continue
            lng_wgs, lat_wgs = to_wgs84(lng_gcj, lat_gcj, cfg.coord_system)
            grid_id = latlng_to_h3(lat_wgs, lng_wgs, cfg.h3_resolution)
            type_text = str(p.get("type") or "")
            typecode = str(p.get("typecode") or "")
            name = str(p.get("name") or "")
            category = classify_poi(type_text, typecode, name)
            seen_ids.add(poi_id)
            counts[(grid_id, category)] += 1
            if cfg.save_unique_poi:
                unique_rows.append({
                    "poi_id": poi_id,
                    "name": name,
                    "category": category,
                    "type": type_text,
                    "typecode": typecode,
                    "lng_gcj": lng_gcj,
                    "lat_gcj": lat_gcj,
                    "lng_wgs": lng_wgs,
                    "lat_wgs": lat_wgs,
                    "grid_id": grid_id,
                    "adname": p.get("adname", ""),
                    "address": p.get("address", ""),
                })
    if cfg.save_unique_poi:
        append_rows_csv(unique_path, unique_rows, [
            "poi_id", "name", "category", "type", "typecode", "lng_gcj", "lat_gcj", "lng_wgs", "lat_wgs", "grid_id", "adname", "address"
        ])

    append_rows_csv(log_path, [{
        "tile_id": tile_id, "depth": depth, "page": "all", "count": count,
        "returned": sum(len(x[1]) for x in all_pages), "status": status, "info": info, "task_status": "done",
    }], ["tile_id", "depth", "page", "count", "returned", "status", "info", "task_status"])
    return counts


def aggregate_unique_poi(unique_path: Path, grid_summary_path: Optional[str], out_dir: Path) -> pd.DataFrame:
    if not unique_path.exists():
        return pd.DataFrame()
    poi = pd.read_csv(unique_path)
    if poi.empty:
        return pd.DataFrame()
    agg = poi.groupby(["grid_id", "category"]).size().unstack(fill_value=0).reset_index()
    for cat in CATEGORY_ORDER:
        if cat not in agg.columns:
            agg[cat] = 0
    # 统一列名，便于模型使用
    rename = {cat: f"poi_{cat}_count" for cat in CATEGORY_ORDER}
    agg = agg.rename(columns=rename)
    agg["poi_total_count"] = agg[[f"poi_{cat}_count" for cat in CATEGORY_ORDER]].sum(axis=1)
    if grid_summary_path:
        gs = pd.read_csv(grid_summary_path)
        if "grid_id" in gs.columns:
            center_cols = [c for c in ["grid_id", "center_lat", "center_lng", "grid_center_lat", "grid_center_lng"] if c in gs.columns]
            base = gs[center_cols].drop_duplicates("grid_id")
            agg = base.merge(agg, on="grid_id", how="left")
            for c in agg.columns:
                if c.startswith("poi_"):
                    agg[c] = agg[c].fillna(0).astype(int)
    agg.to_csv(out_dir / "poi_grid_features.csv", index=False, encoding="utf-8-sig")
    # stage2 点表兼容输出
    stage2 = poi[["poi_id", "name", "lng_gcj", "lat_gcj", "category", "type", "typecode", "grid_id"]].rename(
        columns={"lng_gcj": "lng", "lat_gcj": "lat"}
    )
    stage2.to_csv(out_dir / "poi_for_stage2.csv", index=False, encoding="utf-8-sig")
    poi["category"].value_counts().rename_axis("category").reset_index(name="poi_count").to_csv(
        out_dir / "poi_category_summary.csv", index=False, encoding="utf-8-sig"
    )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="高德 POI 经济型抓取与 H3 聚合")
    parser.add_argument("--amap-key", required=True, help="高德 Web 服务 Key")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--grid-summary", default=None, help="已有 H3 grid_summary CSV，用于限定抓取范围并输出完整网格特征")
    parser.add_argument("--bbox", default=None, help="自定义范围 min_lng,min_lat,max_lng,max_lat；优先级高于 grid-summary")
    parser.add_argument("--tile-size", type=float, default=0.02, help="初始切片大小，单位度。0.02 约 2km 量级")
    parser.add_argument("--offset", type=int, default=25, help="每页返回条数，v3 常用 20/25")
    parser.add_argument("--max-pages", type=int, default=100, help="每个 tile 最大分页数")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="请求间隔")
    parser.add_argument("--max-retry", type=int, default=3, help="失败重试次数")
    parser.add_argument("--max-calls", type=int, default=0, help="最大 API 调用次数预算，0 表示不限制")
    parser.add_argument("--saturation-threshold", type=int, default=850, help="tile 返回 count 超过该值则递归切分")
    parser.add_argument("--min-tile-size", type=float, default=0.003, help="递归切分最小 tile 大小")
    parser.add_argument("--no-recursive-split", action="store_true", help="关闭饱和递归切分")
    parser.add_argument("--coord-system", default="gcj02", choices=["gcj02", "bd09", "wgs84", "raw"], help="高德返回坐标默认为 gcj02")
    parser.add_argument("--h3-resolution", type=int, default=8, help="H3 精度，需与订单网格一致")
    parser.add_argument("--estimate-only", action="store_true", help="只抽样估算调用量，不正式抓取")
    parser.add_argument("--estimate-sample-tiles", type=int, default=40, help="估算模式抽样 tile 数")
    parser.add_argument("--no-save-unique-poi", action="store_true", help="不保存去重 POI 点表，只输出日志。正式研究不建议开启")
    parser.add_argument("--no-resume", action="store_true", help="不使用断点续跑")
    parser.add_argument("--type-codes", default="|".join(DEFAULT_TYPE_CODES), help="高德 POI typecode，用 | 分隔")
    args = parser.parse_args()

    cfg = Config(
        amap_key=args.amap_key,
        out_dir=args.out_dir,
        grid_summary=args.grid_summary,
        tile_size=args.tile_size,
        bbox=args.bbox,
        offset=args.offset,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
        max_retry=args.max_retry,
        max_calls=args.max_calls,
        saturation_threshold=args.saturation_threshold,
        min_tile_size=args.min_tile_size,
        recursive_split=not args.no_recursive_split,
        coord_system=args.coord_system,
        h3_resolution=args.h3_resolution,
        estimate_only=args.estimate_only,
        estimate_sample_tiles=args.estimate_sample_tiles,
        save_unique_poi=not args.no_save_unique_poi,
        resume=not args.no_resume,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.bbox:
        bbox = parse_bbox(cfg.bbox)
    elif cfg.grid_summary:
        bbox = bbox_from_grid_summary(cfg.grid_summary)
    else:
        # 深圳市粗范围。若有订单网格，强烈建议传 --grid-summary 限定运营区。
        bbox = (113.70, 22.30, 114.70, 22.90)

    tiles = generate_tiles(bbox, cfg.tile_size)
    types = args.type_codes

    manifest = {
        "principle": "尽量减少 API 调用；抓取后输出唯一 POI 表和 H3 网格聚合表，后续研究不再调用高德接口。",
        "config": {k: ("***" if k == "amap_key" else v) for k, v in asdict(cfg).items()},
        "bbox": bbox,
        "tile_count_initial": len(tiles),
        "type_codes": types,
        "outputs": ["poi_unique_minimal.csv", "poi_grid_features.csv", "poi_for_stage2.csv", "poi_category_summary.csv", "poi_fetch_log.csv"],
    }
    (out_dir / "poi_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"初始 tile 数：{len(tiles)}；bbox={bbox}；tile_size={cfg.tile_size}")
    print(f"查询 type codes：{types}")
    print("建议先用 --estimate-only 估算调用量，再正式运行。" if not cfg.estimate_only else "当前为估算模式。")

    session = requests.Session()
    if cfg.estimate_only:
        estimate_calls(session, cfg, tiles, types, out_dir)
        return

    unique_path = out_dir / "poi_unique_minimal.csv"
    log_path = out_dir / "poi_fetch_log.csv"
    seen_ids = load_existing_ids(unique_path) if cfg.resume else set()
    done_tiles = load_done_tiles(log_path) if cfg.resume else set()
    print(f"已加载去重 POI ID：{len(seen_ids)}；已完成 tile：{len(done_tiles)}")

    call_counter = {"calls": 0}
    iterator = tqdm(tiles, desc="抓取 POI tile") if tqdm else tiles
    aggregate_counts = Counter()
    for tile in iterator:
        if cfg.max_calls and call_counter["calls"] >= cfg.max_calls:
            print(f"达到 max_calls={cfg.max_calls}，停止。可稍后用同一命令续跑。")
            break
        tile_id = make_tile_id(tile)
        if tile_id in done_tiles:
            continue
        counts = process_tile(session, cfg, tile, types, seen_ids, unique_path, log_path, call_counter)
        aggregate_counts.update(counts)
        time.sleep(cfg.sleep_seconds)

    agg = aggregate_unique_poi(unique_path, cfg.grid_summary, out_dir)
    final_manifest = json.loads((out_dir / "poi_manifest.json").read_text(encoding="utf-8"))
    final_manifest.update({
        "api_calls_this_run": call_counter["calls"],
        "unique_poi_count": int(len(seen_ids)),
        "grid_feature_rows": int(len(agg)) if isinstance(agg, pd.DataFrame) else 0,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    (out_dir / "poi_manifest.json").write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("完成。核心文件：")
    print(f"1. {out_dir / 'poi_unique_minimal.csv'}")
    print(f"2. {out_dir / 'poi_grid_features.csv'}")
    print(f"3. {out_dir / 'poi_for_stage2.csv'}")
    print(f"4. {out_dir / 'poi_fetch_log.csv'}")


if __name__ == "__main__":
    main()
