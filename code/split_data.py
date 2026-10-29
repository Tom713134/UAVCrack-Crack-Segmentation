"""按变焦、按原图分组划分训练 / 选择 / 确认。

同一张航拍切出来的小块带同一个 DJI 时间戳，必须进同一侧。
官方 val 不参与。

默认比例（相对有标注的 1200 张）：
  - 测试集 confirm = 15%
  - 其余 85% 为训练侧；其中再按 SELECT_WITHIN_TRAIN_POOL 划出 select
    （早停 / 扫阈值），不占用测试份额。

数据清洗后（exclude 非空）默认**冻结 confirm**：先用未剔除的原始名单
算出 confirm，再把剔除项从 train/select 中拿掉。这样清洗前后 confirm
组成不变，指标可严格对比。
"""


from __future__ import annotations
import re
from pathlib import Path

ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")
_FRAME = re.compile(r"DJI_(\d+)_(\d+)", re.IGNORECASE)

TEST_RATIO = 0.15
SELECT_WITHIN_TRAIN_POOL = 0.15


def frame_key(name: str) -> str:
    return name.rsplit("_", 2)[0]


def frame_sort_key(name: str) -> tuple:
    match = _FRAME.search(name)
    if match is None:
        return ("", "", name)
    return (match.group(1), match.group(2), name)


def split_zoom(
    names: list[str],
    train_ratio: float | None = None,
    select_ratio: float | None = None,
    *,
    test_ratio: float = TEST_RATIO,
    select_within_train_pool: float = SELECT_WITHIN_TRAIN_POOL,
) -> dict[str, list[str]]:
    """按帧划分到 train / select / confirm。

    默认：confirm≈test_ratio；剩余为训练池，池内再划 select。
    若显式传入 train_ratio/select_ratio，则使用旧的三段 cap 逻辑。
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(frame_key(name), []).append(name)
    ordered = sorted(groups, key=frame_sort_key)
    n = len(names)
    buckets: dict[str, list[str]] = {"train": [], "select": [], "confirm": []}

    if train_ratio is not None or select_ratio is not None:
        tr = 0.6 if train_ratio is None else float(train_ratio)
        sr = 0.2 if select_ratio is None else float(select_ratio)
        train_cap = int(round(n * tr))
        select_cap = int(round(n * sr))
        for key in ordered:
            members = sorted(groups[key])
            if len(buckets["train"]) < train_cap:
                buckets["train"].extend(members)
            elif len(buckets["select"]) < select_cap:
                buckets["select"].extend(members)
            else:
                buckets["confirm"].extend(members)
        return buckets

    confirm_cap = max(1, int(round(n * float(test_ratio))))
    # 从后往前按帧填测试集，前面帧留给训练侧
    confirm_keys: list[str] = []
    n_confirm = 0
    for key in reversed(ordered):
        members = groups[key]
        if n_confirm >= confirm_cap:
            break
        confirm_keys.append(key)
        n_confirm += len(members)
    confirm_key_set = set(confirm_keys)
    for key in confirm_keys:
        buckets["confirm"].extend(sorted(groups[key]))

    train_pool_keys = [k for k in ordered if k not in confirm_key_set]
    pool_n = sum(len(groups[k]) for k in train_pool_keys)
    select_cap = int(round(pool_n * float(select_within_train_pool)))
    n_sel = 0
    for key in train_pool_keys:
        members = sorted(groups[key])
        if n_sel < select_cap:
            buckets["select"].extend(members)
            n_sel += len(members)
        else:
            buckets["train"].extend(members)

    return buckets


def build_split(
    data_root: Path,
    *,
    reference_root: Path | None = None,
    exclude: frozenset[str] | None = None,
) -> dict:
    """构建 train/select/confirm。

    reference_root: 用于**确定划分边界**的原始数据根（默认同 data_root）。
        清洗后仍按原始全量名单划 15%，再剔除 exclude 项 —— confirm 冻结不变。
    exclude: 需要剔除的文件名（仅从 train/select 中移除；如出现在 confirm 会报错）。
    """
    img_root = data_root / "leftImg8bit" / "train"
    ref_root = (reference_root or data_root) / "leftImg8bit" / "train"
    excl = frozenset(exclude or ())
    out = {
        "zooms": {},
        "policy": {
            "test_ratio": TEST_RATIO,
            "train_pool_ratio": 1.0 - TEST_RATIO,
            "select_within_train_pool": SELECT_WITHIN_TRAIN_POOL,
            "note": "confirm=测试15%；其余85%为训练侧，再内划select",
            "reference_root": str(reference_root) if reference_root else None,
            "n_excluded": len(excl),
        },
    }
    for zoom in ZOOMS:
        ref_names = sorted(p.name for p in (ref_root / zoom).glob("*.jpg"))
        parts = split_zoom(ref_names)
        dropped = {"train": [], "select": [], "confirm": []}
        for part in parts:
            kept = []
            for name in parts[part]:
                if name in excl:
                    if part == "confirm":
                        raise RuntimeError(f"剔除项落在 confirm 内，破坏可比性: {zoom}/{name}")
                    dropped[part].append(name)
                    continue
                if (img_root / zoom / name).is_file():
                    kept.append(name)
                else:
                    dropped[part].append(name)
            parts[part] = kept
        out["zooms"][zoom] = {
            "n": {k: len(v) for k, v in parts.items()},
            "files": parts,
            "dropped": {k: len(v) for k, v in dropped.items()},
            "n_reference": len(ref_names),
        }
    return out
