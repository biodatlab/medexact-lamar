"""
Convert model-predicted XML-tagged CSV to evaluation JSON (offset format).
Reads a CSV with columns: file_id, predicted, raw_text
Outputs JSON in the format expected by evaluate.py:
  [{"file_name": "...", "predictions": [{"start_offset": N, "end_offset": M, "category": C}]}]
"""
import argparse
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

CATEGORY_CLASSES = {
    "contact_related": 1,
    "gather_info": 2,
    "define_problem": 3,
    "treatment_goal": 4,
    "drug_decision": 5,
    "therapeutic_procedure": 6,
    "evaluate_result": 7,
    "defer_decision": 8,
    "advice_and_precaution": 9,
}

VALID_TAGS = set(CATEGORY_CLASSES.keys())
TAG_RE = re.compile(r"<(/?)([a-zA-Z_]+)>")


def snap_to_word_boundaries(raw_text, start, end):
    """Snap start backward and end forward to word boundaries."""
    while start > 0 and not raw_text[start - 1].isspace():
        start -= 1
    while end < len(raw_text) and not raw_text[end].isspace():
        end += 1
    return start, end


def extract_spans_from_tags(tag_text):
    """
    Extract (span_text, category, approx_position_ratio) from tagged text.
    Returns spans with their text content and approximate position.
    """
    stack = []
    spans = []

    plain_text = []
    plain_pos = 0
    total_plain_len = len(TAG_RE.sub("", tag_text))
    i = 0
    n = len(tag_text)

    while i < n:
        if tag_text[i] != "<":
            j = tag_text.find("<", i)
            if j == -1:
                j = n
            chunk = tag_text[i:j]
            plain_text.append(chunk)
            plain_pos += len(chunk)
            i = j
            continue

        m = TAG_RE.match(tag_text, i)
        if not m:
            plain_text.append(tag_text[i])
            plain_pos += 1
            i += 1
            continue

        is_close = m.group(1) == "/"
        tag = m.group(2)

        if tag not in VALID_TAGS:
            i = m.end()
            continue

        if not is_close:
            stack.append((tag, plain_pos))
        else:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == tag:
                    _, start = stack.pop(k)
                    plain_joined = "".join(plain_text)
                    span_text = plain_joined[start:plain_pos]

                    pos_ratio = start / max(total_plain_len, 1)

                    spans.append({
                        "text": span_text,
                        "category": CATEGORY_CLASSES[tag],
                        "pred_start": start,
                        "pred_end": plain_pos,
                        "pos_ratio": pos_ratio,
                    })
                    break

        i = m.end()

    return spans


def _pick_best(matches, raw_text, expected_ratio):
    """Pick the match closest to the expected position ratio."""
    if len(matches) == 1:
        return matches[0]
    raw_len = max(len(raw_text), 1)
    return min(matches, key=lambda m: abs(m[0] / raw_len - expected_ratio))


def _find_all(haystack, needle, start=0):
    """Find all occurrences of needle in haystack."""
    matches = []
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(needle)))
        start = idx + 1
    return matches


def find_span_in_raw(span_text, raw_text, expected_ratio=0.5):
    """
    Find the best matching position of span_text in raw_text.
    Returns (start_offset, end_offset, match_type) or (None, None, reason).
    match_type is one of: "exact", "normalized", "case_insensitive", "prefix",
                          "suffix", "fuzzy".
    reason is one of: "empty_span", "model_typo", "not_in_text", "truncated".
    """
    if not span_text or not span_text.strip():
        return None, None, "empty_span"

    # 1. Exact match
    matches = _find_all(raw_text, span_text)
    if matches:
        best = _pick_best(matches, raw_text, expected_ratio)
        return best[0], best[1], "exact"

    # 2. Whitespace-normalized match
    span_normalized = " ".join(span_text.split())
    raw_normalized = " ".join(raw_text.split())
    if span_normalized != span_text:
        matches = _find_all(raw_text, span_normalized)
        if matches:
            best = _pick_best(matches, raw_text, expected_ratio)
            return best[0], best[1], "normalized"

    if span_normalized:
        idx = raw_normalized.find(span_normalized)
        if idx != -1:
            orig_idx = _map_normalized_to_original(raw_text, raw_normalized, idx)
            if orig_idx is not None:
                end_idx = _find_end_in_original(raw_text, orig_idx, span_normalized)
                return orig_idx, end_idx, "normalized"

    # 3. Case-insensitive match
    span_lower = span_text.lower()
    raw_lower = raw_text.lower()
    matches = _find_all(raw_lower, span_lower)
    if matches:
        best = _pick_best(matches, raw_text, expected_ratio)
        return best[0], best[1], "case_insensitive"

    # 4. Prefix matching (first 50 chars)
    if len(span_text) > 20:
        prefix = span_text[:50]
        idx = raw_text.find(prefix)
        if idx != -1:
            end = min(idx + len(span_text), len(raw_text))
            return idx, end, "prefix"
        idx = raw_lower.find(prefix.lower())
        if idx != -1:
            end = min(idx + len(span_text), len(raw_text))
            return idx, end, "prefix"

    # 5. Suffix matching (last 50 chars)
    if len(span_text) > 20:
        suffix = span_text[-50:]
        idx = raw_text.find(suffix)
        if idx != -1:
            start = max(idx - (len(span_text) - len(suffix)), 0)
            return start, idx + len(suffix), "suffix"

    # 6. Fuzzy sliding-window match
    fuzzy_result = _fuzzy_match(span_text, raw_text, expected_ratio, threshold=0.85)
    if fuzzy_result:
        return fuzzy_result[0], fuzzy_result[1], "fuzzy"

    reason = _classify_failure(span_text, raw_text)
    return None, None, reason


def _map_normalized_to_original(raw_text, raw_normalized, norm_idx):
    """Map an index in normalized text back to the original text."""
    norm_pos = 0
    orig_pos = 0
    while norm_pos < norm_idx and orig_pos < len(raw_text):
        if raw_text[orig_pos].isspace():
            while orig_pos < len(raw_text) and raw_text[orig_pos].isspace():
                orig_pos += 1
            norm_pos += 1
        else:
            orig_pos += 1
            norm_pos += 1
    return orig_pos if norm_pos == norm_idx else None


def _find_end_in_original(raw_text, start_idx, normalized_span):
    """Find end offset in original text that covers the normalized span."""
    norm_pos = 0
    orig_pos = start_idx
    while norm_pos < len(normalized_span) and orig_pos < len(raw_text):
        if normalized_span[norm_pos] == ' ' and raw_text[orig_pos].isspace():
            while orig_pos < len(raw_text) and raw_text[orig_pos].isspace():
                orig_pos += 1
            norm_pos += 1
        else:
            orig_pos += 1
            norm_pos += 1
    return orig_pos


def _fuzzy_match(span_text, raw_text, expected_ratio, threshold=0.85):
    """Fuzzy sliding-window match. Returns (start, end) or None."""
    span_len = len(span_text)
    if span_len < 5:
        return None

    raw_len = len(raw_text)
    if raw_len == 0:
        return None

    min_win = max(int(span_len * 0.8), 1)
    max_win = min(int(span_len * 1.2), raw_len)

    best_ratio = 0.0
    best_match = None

    step = max(1, span_len // 10)

    for win_size in [span_len, min_win, max_win]:
        if win_size > raw_len:
            continue
        for i in range(0, raw_len - win_size + 1, step):
            candidate = raw_text[i:i + win_size]
            ratio = SequenceMatcher(None, span_text, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = (i, i + win_size)

    # Refine around best match with step=1
    if best_match and best_ratio >= threshold * 0.9:
        refine_start = max(0, best_match[0] - step)
        refine_end = min(raw_len, best_match[0] + step)
        for win_size in [span_len, min_win, max_win]:
            if win_size > raw_len:
                continue
            for i in range(refine_start, min(refine_end, raw_len - win_size + 1)):
                candidate = raw_text[i:i + win_size]
                ratio = SequenceMatcher(None, span_text, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (i, i + win_size)

    if best_ratio >= threshold and best_match:
        return best_match
    return None


def _classify_failure(span_text, raw_text):
    """Classify why a span failed to match."""
    span_len = len(span_text)
    if span_len >= 5:
        result = _fuzzy_match(span_text, raw_text, 0.5, threshold=0.6)
        if result:
            return "model_typo"

    if span_len > 10:
        prefix = span_text[:max(10, span_len // 3)]
        if raw_text.find(prefix) != -1:
            return "truncated"

    return "not_in_text"


def convert_predictions(csv_path, out_path, keep_decision=False):
    df = pd.read_csv(csv_path)
    results = []
    match_types = {"exact": 0, "normalized": 0, "case_insensitive": 0,
                   "prefix": 0, "suffix": 0, "fuzzy": 0}
    fail_reasons = {"empty_span": 0, "model_typo": 0, "not_in_text": 0, "truncated": 0}
    total = 0
    dropped_details = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {Path(csv_path).name}"):
        file_id = str(row["file_id"])
        tag_pred = str(row["predicted"])
        raw_text = str(row["raw_text"])

        pred_spans = extract_spans_from_tags(tag_pred)

        aligned_spans = []
        for span in pred_spans:
            total += 1

            start, end, match_info = find_span_in_raw(
                span["text"],
                raw_text,
                expected_ratio=span["pos_ratio"]
            )

            if start is not None:
                match_types[match_info] = match_types.get(match_info, 0) + 1

                span_dict = {
                    "start_offset": start,
                    "end_offset": end,
                    "category": span["category"],
                }
                if keep_decision:
                    snapped_start, snapped_end = snap_to_word_boundaries(raw_text, start, end)
                    span_dict["decision"] = raw_text[snapped_start:snapped_end]
                aligned_spans.append(span_dict)
            else:
                fail_reasons[match_info] = fail_reasons.get(match_info, 0) + 1
                closest = ""
                if match_info == "model_typo":
                    result = _fuzzy_match(span["text"], raw_text, span["pos_ratio"], threshold=0.6)
                    if result:
                        closest = raw_text[result[0]:result[1]]
                dropped_details.append({
                    "file_id": file_id,
                    "category": span["category"],
                    "reason": match_info,
                    "span_text": span["text"][:80],
                    "closest": closest[:80] if closest else "",
                })

        aligned_spans.sort(key=lambda x: (x["start_offset"], x["end_offset"]))

        results.append({
            "file_name": file_id,
            "predictions": aligned_spans,
        })

    # Print alignment stats
    if total > 0:
        matched = sum(match_types.values())
        failed = sum(fail_reasons.values())
        print(f"\nAlignment stats ({total} spans):")
        print(f"  Matched: {matched} ({matched/total*100:.1f}%)")
        for mtype, count in match_types.items():
            if count > 0:
                print(f"    {mtype:20s}: {count}")
        print(f"  Dropped: {failed} ({failed/total*100:.1f}%)")
        for reason, count in fail_reasons.items():
            if count > 0:
                print(f"    {reason:20s}: {count}")

    if dropped_details:
        print(f"\nDropped spans ({len(dropped_details)}):")
        for d in dropped_details:
            line = f"  [{d['file_id']}] cat={d['category']} reason={d['reason']}: \"{d['span_text']}\""
            if d["closest"]:
                line += f" → closest: \"{d['closest']}\""
            print(line)

        dropped_csv = str(Path(out_path).with_name(
            Path(out_path).stem + "_dropped.csv"))
        pd.DataFrame(dropped_details).to_csv(dropped_csv, index=False)
        print(f"  Dropped spans saved to: {dropped_csv}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def convert_batch(input_dir, keep_decision=False):
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No .csv files found in: {input_dir}")
        return
    for csv_path in csv_files:
        out_path = csv_path.with_suffix(".json")
        convert_predictions(csv_path, out_path, keep_decision=keep_decision)
        print("Saved to:", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert model-predicted XML-tagged CSV to evaluation JSON (offset format)."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input CSV path, or directory when --batch is set")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: input with .json suffix). Ignored in batch mode.")
    parser.add_argument("--batch", action="store_true",
                        help="Treat --input as a directory and process all CSVs")
    parser.add_argument("--keep_decision", action="store_true",
                        help="Include word-boundary-snapped decision text in output")
    args = parser.parse_args()

    if args.batch:
        convert_batch(args.input, keep_decision=args.keep_decision)
    else:
        out_path = args.output or args.input.with_suffix(".json")
        convert_predictions(args.input, out_path, keep_decision=args.keep_decision)
        print("Saved to:", out_path)


if __name__ == "__main__":
    main()
