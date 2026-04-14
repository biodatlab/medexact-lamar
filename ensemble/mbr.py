import json
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd


pred_list =[
    'val_model1_dft.json',
    'val_model2_dft_grpo.json',
    'val_model3_dft_unlabelp10.json'
]

# sample_id = '29043_173556_32734'
sample_id = float("nan")

OUTPUT_JSON_SINGLE = None   # e.g. "single_doc_mbr.json" or None for auto name
OUTPUT_JSON_ALL = "mbr_merged_3_models.json"


# =========================================================
# Validation / basic helpers
# =========================================================
VALID_CATEGORIES = set(range(1, 10))


def validate_and_normalize_predictions(indexed_predictions: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
    """
    Validate predictions for one document.
    Input:
        {
            model_idx: [
                {"start_offset": int, "end_offset": int, "category": int},
                ...
            ],
            ...
        }

    Output:
        Same structure, but invalid spans removed and model_idx added to each span.
    """
    cleaned = {}

    for model_idx, preds in indexed_predictions.items():
        valid_spans = []

        if preds is None:
            cleaned[model_idx] = []
            continue

        for span in preds:
            if not isinstance(span, dict):
                continue

            if not all(k in span for k in ("start_offset", "end_offset", "category")):
                continue

            s = span["start_offset"]
            e = span["end_offset"]
            c = span["category"]

            if not isinstance(s, int) or not isinstance(e, int) or not isinstance(c, int):
                continue
            if c not in VALID_CATEGORIES:
                continue
            if s < 0 or e <= s:
                continue

            valid_spans.append({
                "start_offset": s,
                "end_offset": e,
                "category": c,
                "model_idx": model_idx,
            })

        cleaned[model_idx] = valid_spans

    return cleaned


def iou_span(a: Dict, b: Dict) -> float:
    """
    Character-level IoU for two spans.
    """
    inter = max(0, min(a["end_offset"], b["end_offset"]) - max(a["start_offset"], b["start_offset"]))
    if inter == 0:
        return 0.0

    union = max(a["end_offset"], b["end_offset"]) - min(a["start_offset"], b["start_offset"])
    return inter / union if union > 0 else 0.0


def overlaps(a: Dict, b: Dict, min_iou: float = 0.1) -> bool:
    """
    Overlap decision within same category only.
    """
    if a["category"] != b["category"]:
        return False
    return iou_span(a, b) >= min_iou


# =========================================================
# MBR-like model agreement
# =========================================================
def greedy_soft_match_score(spans_a: List[Dict], spans_b: List[Dict], min_iou: float = 0.1) -> float:
    """
    Soft F1-like similarity between two span sets from the same category.

    Greedy matching by highest IoU.
    """
    if not spans_a and not spans_b:
        return 1.0
    if not spans_a or not spans_b:
        return 0.0

    candidates = []
    for i, a in enumerate(spans_a):
        for j, b in enumerate(spans_b):
            score = iou_span(a, b)
            if score >= min_iou:
                candidates.append((score, i, j))

    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_a = set()
    matched_b = set()
    total_match_score = 0.0

    for score, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        total_match_score += score

    precision = total_match_score / len(spans_a) if spans_a else 0.0
    recall = total_match_score / len(spans_b) if spans_b else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_model_weights_for_category(
    category_spans_by_model: Dict[int, List[Dict]],
    min_iou: float = 0.1
) -> Dict[int, float]:
    """
    Compute MBR-like reliability weights for one category in one document.
    Each model gets average soft agreement with the other models.
    """
    model_ids = sorted(category_spans_by_model.keys())
    weights = {}

    if len(model_ids) == 1:
        return {model_ids[0]: 1.0}

    for m in model_ids:
        sims = []
        for n in model_ids:
            if m == n:
                continue
            sim = greedy_soft_match_score(
                category_spans_by_model[m],
                category_spans_by_model[n],
                min_iou=min_iou
            )
            sims.append(sim)

        weights[m] = sum(sims) / len(sims) if sims else 0.0

    total = sum(weights.values())
    if total > 0:
        weights = {m: w / total for m, w in weights.items()}
    else:
        uniform = 1.0 / len(model_ids)
        weights = {m: uniform for m in model_ids}

    return weights


# =========================================================
# Clustering + refinement
# =========================================================
def build_overlap_clusters(spans: List[Dict], min_iou: float = 0.1) -> List[List[Dict]]:
    """
    Build connected components over overlap graph.
    Assumes all spans are same category.
    """
    n = len(spans)
    if n == 0:
        return []

    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        cluster_idx = []

        while stack:
            cur = stack.pop()
            cluster_idx.append(cur)

            for j in range(n):
                if visited[j]:
                    continue
                if overlaps(spans[cur], spans[j], min_iou=min_iou):
                    visited[j] = True
                    stack.append(j)

        clusters.append([spans[k] for k in cluster_idx])

    return clusters


def cluster_support(cluster: List[Dict], model_weights: Dict[int, float]) -> float:
    """
    Weighted support from distinct contributing models.
    """
    supporting_models = set(span["model_idx"] for span in cluster)
    return sum(model_weights.get(m, 0.0) for m in supporting_models)


def weighted_boundary_vote(cluster: List[Dict], model_weights: Dict[int, float]) -> Tuple[int, int]:
    """
    Refine a cluster into one final boundary.
    Final boundary is always one of the observed boundaries.
    """
    boundary_scores = defaultdict(float)
    start_scores = defaultdict(float)
    end_scores = defaultdict(float)

    for span in cluster:
        w = model_weights.get(span["model_idx"], 0.0)
        key = (span["start_offset"], span["end_offset"])
        boundary_scores[key] += w
        start_scores[span["start_offset"]] += w
        end_scores[span["end_offset"]] += w

    best_boundary, best_score = max(
        boundary_scores.items(),
        key=lambda x: (x[1], -(x[0][1] - x[0][0]))
    )

    best_start = max(start_scores.items(), key=lambda x: (x[1], -x[0]))[0]
    best_end = max(end_scores.items(), key=lambda x: (x[1], x[0]))[0]

    observed_boundaries = list(boundary_scores.keys())
    candidate_boundary = min(
        observed_boundaries,
        key=lambda x: (abs(x[0] - best_start) + abs(x[1] - best_end), -boundary_scores[x])
    )

    if boundary_scores[candidate_boundary] > best_score:
        return candidate_boundary

    return best_boundary


def deduplicate_final_spans(spans: List[Dict], min_iou_same_cat: float = 0.8) -> List[Dict]:
    """
    Deduplicate nearly identical spans within the same category.
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x["category"], x["start_offset"], x["end_offset"]))
    kept = []

    for span in spans:
        should_add = True
        replace_idx = None

        for i, prev in enumerate(kept):
            if prev["category"] != span["category"]:
                continue

            iou = iou_span(prev, span)
            if iou >= min_iou_same_cat:
                prev_len = prev["end_offset"] - prev["start_offset"]
                cur_len = span["end_offset"] - span["start_offset"]
                if cur_len > prev_len:
                    replace_idx = i
                should_add = False
                break

        if replace_idx is not None:
            kept[replace_idx] = span
        elif should_add:
            kept.append(span)

    return kept


# =========================================================
# Single-document pipeline
# =========================================================
def mbr_cluster_and_refine_single_doc(
    indexed_predictions: Dict[int, List[Dict]],
    min_cluster_iou: float = 0.1,
    min_pairwise_iou: float = 0.1,
    min_cluster_support: float = 0.34,
    min_iou_same_cat: float = 0.8,
    verbose: bool = False,
) -> Dict:
    """
    Process one document:
      1) validate spans
      2) compute per-category model weights
      3) pool spans per category
      4) cluster by overlap
      5) refine cluster boundaries
      6) keep sufficiently supported clusters
    """
    cleaned = validate_and_normalize_predictions(indexed_predictions)
    all_model_ids = sorted(cleaned.keys())

    by_category_by_model = {c: {m: [] for m in all_model_ids} for c in VALID_CATEGORIES}
    for model_idx, spans in cleaned.items():
        for span in spans:
            by_category_by_model[span["category"]][model_idx].append(span)

    final_predictions = []
    debug_info = {
        "model_weights_by_category": {},
        "clusters_by_category": {},
    }

    for category in sorted(VALID_CATEGORIES):
        category_spans_by_model = by_category_by_model[category]

        model_weights = compute_model_weights_for_category(
            category_spans_by_model=category_spans_by_model,
            min_iou=min_pairwise_iou
        )
        debug_info["model_weights_by_category"][category] = model_weights

        pooled_spans = []
        for model_idx in all_model_ids:
            pooled_spans.extend(category_spans_by_model[model_idx])

        if not pooled_spans:
            debug_info["clusters_by_category"][category] = []
            continue

        clusters = build_overlap_clusters(pooled_spans, min_iou=min_cluster_iou)
        cluster_records = []

        for cluster in clusters:
            support = cluster_support(cluster, model_weights)
            refined_start, refined_end = weighted_boundary_vote(cluster, model_weights)

            record = {
                "category": category,
                "support": support,
                "cluster_size": len(cluster),
                "models": sorted(set(x["model_idx"] for x in cluster)),
                "members": sorted(
                    [
                        {
                            "start_offset": x["start_offset"],
                            "end_offset": x["end_offset"],
                            "category": x["category"],
                            "model_idx": x["model_idx"],
                        }
                        for x in cluster
                    ],
                    key=lambda z: (z["start_offset"], z["end_offset"], z["model_idx"])
                ),
                "refined_span": {
                    "start_offset": refined_start,
                    "end_offset": refined_end,
                    "category": category,
                }
            }
            cluster_records.append(record)

            if support >= min_cluster_support:
                final_predictions.append({
                    "start_offset": refined_start,
                    "end_offset": refined_end,
                    "category": category,
                })

        debug_info["clusters_by_category"][category] = cluster_records

    final_predictions = deduplicate_final_spans(final_predictions, min_iou_same_cat=min_iou_same_cat)
    final_predictions = sorted(
        final_predictions,
        key=lambda x: (x["start_offset"], x["end_offset"], x["category"])
    )

    if verbose:
        print("=" * 80)
        print("Final predictions")
        for span in final_predictions:
            print(span)

        print("\n" + "=" * 80)
        print("Model weights by category")
        for cat in sorted(debug_info["model_weights_by_category"].keys()):
            print(f"Category {cat}: {debug_info['model_weights_by_category'][cat]}")

    return {
        "final_predictions": final_predictions,
        "debug_info": debug_info,
        "cleaned_predictions": cleaned,
    }


# =========================================================
# Input loading
# =========================================================
def load_predictions_for_one_or_all_docs(pred_list: List[str], sample_id):
    """
    Returns:
      process_all: bool
      payload:
        - if process_all is False:
            {model_idx: predictions_for_sample_id}
        - if process_all is True:
            {file_name: {model_idx: predictions_for_that_doc}}
    """
    process_all = pd.isna(sample_id)

    if process_all:
        all_indexed_predictions = defaultdict(dict)

        for idx, json_path in enumerate(pred_list):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                file_name = item.get("file_name")
                preds = item.get("predictions", [])

                if file_name is None:
                    continue

                all_indexed_predictions[file_name][idx] = preds

        return True, all_indexed_predictions

    else:
        indexed_predictions = {}

        for idx, json_path in enumerate(pred_list):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            found = False
            for item in data:
                if item.get("file_name") == sample_id:
                    indexed_predictions[idx] = item.get("predictions", [])
                    found = True
                    break

            if not found:
                indexed_predictions[idx] = []

        return False, indexed_predictions


# =========================================================
# Main run
# =========================================================
process_all, payload = load_predictions_for_one_or_all_docs(pred_list, sample_id)

if process_all:
    all_results = []

    for file_name, indexed_predictions in payload.items():
        result = mbr_cluster_and_refine_single_doc(
            indexed_predictions=indexed_predictions,
            min_cluster_iou=0.1,
            min_pairwise_iou=0.1,
            min_cluster_support=0.34,
            verbose=False,
        )

        all_results.append({
            "file_name": file_name,
            "predictions": result["final_predictions"]
        })

    all_results = sorted(all_results, key=lambda x: str(x["file_name"]))

    # Final saved JSON is always a list of dict
    with open(OUTPUT_JSON_ALL, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_results)} documents to: {OUTPUT_JSON_ALL}")

else:
    result = mbr_cluster_and_refine_single_doc(
        indexed_predictions=payload,
        min_cluster_iou=0.1,
        min_pairwise_iou=0.1,
        min_cluster_support=0.34,
        verbose=True,
    )

    final_output = [
        {
            "file_name": sample_id,
            "predictions": result["final_predictions"]
        }
    ]

    output_path = OUTPUT_JSON_SINGLE or f"{sample_id}_mbr_merged.json"

    # Final saved JSON is always a list of dict
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"Saved 1 document to: {output_path}")