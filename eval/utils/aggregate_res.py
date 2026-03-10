import os
import json
import math
from typing import Dict, List, Any
from pathlib import Path


KEEP_KEYS = ["object coverage", "attribute score", "relation score", "Unified score"]


def mean_and_ci95(values: List[float]):
    """
    给定一组数值，计算平均数和 95% 置信区间误差（即 mean ± error 中的 error）。
    这里使用常见做法：error = 1.96 * (std / sqrt(n)) （大样本近似正态）。
    """
    n = len(values)
    if n == 0:
        return None, None

    mean_val = sum(values) / n
    if n == 1:
        # 只有一个样本时，无法估计标准差，误差设为 0
        return mean_val, 0.0

    # 样本标准差（无偏），除以 (n-1)
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)          # standard error
    error = 1.96 * se               # 95% CI for large n (normal approx)

    return mean_val, error


def _is_number(x: Any):
    try:
        float(x)
        return True
    except Exception:
        return False


def _extract_scene_scores(score_dict: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    从 multi-scene 的 score_dict["scenes"] 提取每个 scene 的四项分数:
      return: {scene_name: {k: float}}
    若不是 multi-scene/缺失则返回 {}。
    """
    scenes_out: Dict[str, Dict[str, float]] = {}
    scenes = score_dict.get("scenes", None)
    if not isinstance(scenes, dict):
        return scenes_out

    for scene_name, sd in scenes.items():
        if not isinstance(sd, dict):
            continue
        one = {}
        for k in KEEP_KEYS:
            v = sd.get(k, None)
            if _is_number(v):
                one[k] = float(v)
        if one:
            scenes_out[str(scene_name)] = one
    return scenes_out


def _extract_final_weighted_scores(score_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    从 multi-scene 的 score_dict["final_weighted"] 提取四项分数。
    若缺失则返回 {}。
    """
    fw = score_dict.get("final_weighted", None)
    if not isinstance(fw, dict):
        return {}
    out = {}
    for k in KEEP_KEYS:
        v = fw.get(k, None)
        if _is_number(v):
            out[k] = float(v)
    return out


def _flatten_scores_for_time_agg(score_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    兼容：
    - single-scene：{"object coverage":..., "Unified score":...} -> 只取数值
    - multi-scene：{"scenes":{...}, "final_weighted":{...}} -> 扁平化为：
        final_weighted.{k}
        scenes.{scene}.{k}
    """
    flat: Dict[str, float] = {}

    # multi-scene
    if isinstance(score_dict.get("final_weighted"), dict) or isinstance(score_dict.get("scenes"), dict):
        fw = score_dict.get("final_weighted", {})
        if isinstance(fw, dict):
            for k, v in fw.items():
                if _is_number(v):
                    flat[f"final_weighted.{k}"] = float(v)

        scenes = score_dict.get("scenes", {})
        if isinstance(scenes, dict):
            for s, sd in scenes.items():
                if isinstance(sd, dict):
                    for k, v in sd.items():
                        if _is_number(v):
                            flat[f"scenes.{s}.{k}"] = float(v)
        return flat

    # single-scene：只收集数值字段（跳过 mode 等字符串）
    for k, v in score_dict.items():
        if _is_number(v):
            flat[k] = float(v)
    return flat


def aggregate_mutil_time_results(input_dir: str):
    """
    对所有 aggregated_mutil_woker_result_*.json 的第一个 dict 做 mean±CI 聚合。
    single-scene：保持不变
    multi-scene：会包含 scenes.*.* 与 final_weighted.* 的统计
    """
    values_per_key: Dict[str, List[float]] = {}

    json_files = list(Path(input_dir).rglob("aggregated_mutil_woker_result_*.json"))
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skip unresolved JSON file: {json_file}")
                continue

        if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], dict):
            print(f"Skip file with structures that do not meet the requirements: {json_file}")
            continue

        flat = _flatten_scores_for_time_agg(data[0])
        for k, v in flat.items():
            values_per_key.setdefault(k, []).append(float(v))

    print(f"==========Final results==========")
    result = {}
    for k, vals in values_per_key.items():
        mean_val, error = mean_and_ci95(vals)
        if mean_val is None:
            continue
        result[k] = {"mean": mean_val, "error": error}
        print(f"{k}: {result[k]}")

    output_json = os.path.join(input_dir, "aggregated_mutil_time_result.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    print(f'results saved to {output_json}')


def aggregate_mutil_woker_results(root_dir: str, target_name: str):
    """
    聚合 root_dir 下递归找到的所有 target_name（如 result_0.json）。
    每个 result: [score_dict, sample_dict]

    single-scene：输出顶层四项
    multi-scene：输出仅保留
      - final_weighted: 四项（按样本数加权聚合）
      - scenes: 各scene四项（按样本数加权聚合）
    """
    paths = list(Path(root_dir).rglob(target_name))
    if not paths:
        raise FileNotFoundError(f"No files named {target_name} under {root_dir}")

    sum_w = 0.0
    merged_samples = {}

    # single-scene/top-level accumulator（仅在最终判定为 single-scene 时使用）
    sum_scores_top = {k: 0.0 for k in KEEP_KEYS}

    # multi-scene accumulators
    sum_scores_fw = {k: 0.0 for k in KEEP_KEYS}                  # final_weighted
    sum_scores_scenes: Dict[str, Dict[str, float]] = {}          # scenes[scene][k] sum
    any_multi_scene = False

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not (isinstance(data, list) and len(data) >= 2 and isinstance(data[0], dict) and isinstance(data[1], dict)):
            raise ValueError(f"Bad format: {p} (expect [dict, dict])")

        score_dict, sample_dict = data[0], data[1]
        w = float(len(sample_dict))
        if w <= 0:
            continue

        # 判定 multi-scene：存在 final_weighted 或 scenes 的 dict
        is_multi = isinstance(score_dict.get("final_weighted"), dict) or isinstance(score_dict.get("scenes"), dict)
        if is_multi:
            any_multi_scene = True

        # single-scene 顶层四项累加（最终若判定为 multi-scene 将不会使用）
        for k in KEEP_KEYS:
            v = score_dict.get(k, 0.0)
            if _is_number(v):
                sum_scores_top[k] += float(v) * w

        # multi-scene：final_weighted 四项累加
        fw = _extract_final_weighted_scores(score_dict)
        if fw:
            for k in KEEP_KEYS:
                if k in fw:
                    sum_scores_fw[k] += float(fw[k]) * w

        # multi-scene：各 scenes 四项累加
        scenes = _extract_scene_scores(score_dict)
        if scenes:
            for scene_name, sd in scenes.items():
                if scene_name not in sum_scores_scenes:
                    sum_scores_scenes[scene_name] = {k: 0.0 for k in KEEP_KEYS}
                for k in KEEP_KEYS:
                    if k in sd:
                        sum_scores_scenes[scene_name][k] += float(sd[k]) * w

        sum_w += w
        merged_samples.update(sample_dict)

    if sum_w == 0:
        raise ValueError("Total sample count is 0 (all second dicts are empty).")

    # 组装输出 score_dict
    if any_multi_scene:
        final_weighted = {k: v / sum_w for k, v in sum_scores_fw.items()}
        scenes_out: Dict[str, Dict[str, float]] = {}
        for scene_name, ksums in sum_scores_scenes.items():
            scenes_out[scene_name] = {k: v / sum_w for k, v in ksums.items()}

        out_score_dict: Dict[str, Any] = {
            "final_weighted": final_weighted,
            "scenes": scenes_out
        }
    else:
        out_score_dict = {k: v / sum_w for k, v in sum_scores_top.items()}

    out = [out_score_dict, merged_samples]
    
    save_path = Path(os.path.join(root_dir, f'aggregated_mutil_woker_{target_name}'))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'results saved to {str(save_path)}')
    return out
