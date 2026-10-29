"""最终版推理阈值：分变焦 τ_z（来自 cal_safe_x5 select 上标定）。

定稿 + 分变焦 τ_z 为当前最终版本；后续所有实验统一用它评测，
避免"统一 0.2"与"分变焦"两套口径混用。
"""

from __future__ import annotations

import json
from pathlib import Path

# cal_safe_x5_result.json -> tau_z_meta.tau_z
DEFAULT_TAU_Z: dict[str, float] = {
    "UAV-CrackX4": 0.18,
    "UAV-CrackX8": 0.20,
    "UAV-CrackX16": 0.10,
}

# 参考基线（原数据 1200 张 / confirm 193 张，CAL+Dice×5）
BASELINE_TAU_GLOBAL = {"Crack IoU": 64.5645, "Crack Precision": 77.3701, "Crack Recall": 79.5957}
BASELINE_TAU_Z = {"Crack IoU": 64.6226, "Crack Precision": 77.0290, "Crack Recall": 80.0490}


def resolve_tau(threshold: float | dict[str, float] | None, zoom: str) -> float:
    if threshold is None:
        threshold = DEFAULT_TAU_Z
    if isinstance(threshold, dict):
        if zoom in threshold:
            return float(threshold[zoom])
        short = {"UAV-CrackX4": "X4", "UAV-CrackX8": "X8", "UAV-CrackX16": "X16"}.get(zoom, zoom)
        if short in threshold:
            return float(threshold[short])
        return float(threshold.get("all", 0.2))
    return float(threshold)


def load_tau_z(path: Path | str | None) -> dict[str, float]:
    """从 result json 里取 tau_z_meta.tau_z；失败则回落默认值。"""
    if path is None:
        return dict(DEFAULT_TAU_Z)
    p = Path(path)
    if not p.is_file():
        return dict(DEFAULT_TAU_Z)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_TAU_Z)
    meta = payload.get("tau_z_meta") or {}
    tau = meta.get("tau_z") or {}
    if not tau:
        return dict(DEFAULT_TAU_Z)
    out = dict(DEFAULT_TAU_Z)
    for k, v in tau.items():
        out[str(k)] = float(v)
    return out
