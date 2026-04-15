"""
Convert gold JSON annotations + raw text into XML-tagged text CSVs for SFT training.

Produces train.csv and val.csv with columns: file_id, raw_text, gold_span, gold_offsets
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CATEGORY_CLASSES = {
    1: "contact_related",
    2: "gather_info",
    3: "define_problem",
    4: "treatment_goal",
    5: "drug_decision",
    6: "therapeutic_procedure",
    7: "evaluate_result",
    8: "defer_decision",
    9: "advice_and_precaution",
}

VALID_CATEGORY_IDS = set(CATEGORY_CLASSES.keys())

#Class with higher prevalence is more inner, to break ties in sorting spans with same offsets
_INNER_ORDER = [3, 5, 7, 1, 6, 9, 2, 4, 8] 
CATEGORY_INNER_ORDER = {cat: i for i, cat in enumerate(_INNER_ORDER)}

_CATEGORY_RE = re.compile(r"Category\s*(\d+)")


def parse_category(category_str: str) -> int:
    """Extract category ID from strings like 'Category 3: Defining problem'."""
    m = _CATEGORY_RE.match(category_str or "")
    if not m:
        return 0
    cat = int(m.group(1))
    return cat if cat in VALID_CATEGORY_IDS else 0


def read_split_ids(split_path: Path) -> list[str]:
    """Read split file lines, stripping whitespace and skipping empties."""
    with split_path.open("r", encoding="utf-8") as f:
        return sorted([line.strip() for line in f if line.strip()])


def convert_xml_from_loaded(file_id: str, raw_text: str, gold_data: dict) -> dict:
    """Convert full raw_text to span/tag XML using GLOBAL offsets from gold_data."""
    text_len = len(raw_text)

    # Collect valid annotations
    anns = []
    for ann in gold_data.get("annotations", []):
        try:
            start = int(ann["start_offset"])
            end = int(ann["end_offset"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (0 <= start < end <= text_len):
            continue

        cat = parse_category(ann.get("category", ""))
        if cat in VALID_CATEGORY_IDS:
            class_name = CATEGORY_CLASSES.get(cat, "unknown")
            anns.append((start, end, class_name, cat))

    # Sort annotations: start asc, end desc (outer first), outermost category first
    anns.sort(key=lambda x: (x[0], -x[1], -CATEGORY_INNER_ORDER.get(x[3], 4)))

    # Build XML tag events referencing original text positions
    # type_order: 0=close fires before 1=open at the same position.
    # tiebreak for closes: -start  → later-opened span closes first (LIFO)
    # tiebreak for opens:  -end    → longer (outer) span opens first
    events = []
    for i, (start, end, class_name, _cat) in enumerate(anns):
        events.append((start, 1, -end,   i,  f'<{class_name}>'))
        events.append((end,   0, -start, -i, f'</{class_name}>'))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    # Single left-to-right pass over the original raw_text
    xml_parts = []
    prev = 0
    for pos, _type_order, _tiebreak, _idx, tag in events:
        xml_parts.append(raw_text[prev:pos])
        xml_parts.append(tag)
        prev = pos
    xml_parts.append(raw_text[prev:])

    xml_span = ''.join(xml_parts)

    gold_offsets = [{'start_offset': start, 'end_offset': end, 'category': cat} for start, end, _cls, cat in anns]
    return {
        "file_id": file_id,
        "raw_text": raw_text,
        "gold_span": xml_span,
        "gold_offsets": json.dumps(gold_offsets)
    }


def process_split(split_ids: list[str], split_name: str,
                  gold_data_dir: Path, raw_text_df: pd.DataFrame) -> pd.DataFrame:
    """Process a split list of JSON filenames into a DataFrame."""
    records = []

    for name in tqdm(split_ids, desc=f"Processing {split_name}", total=len(split_ids)):
        json_path = gold_data_dir / name
        file_id = json_path.stem
        text_row = raw_text_df[raw_text_df['file_id'] == file_id]
        raw_text = text_row.iloc[0]['raw_text'] if not text_row.empty else ""
        if not raw_text:
            print('Missing raw text for file_id:', file_id)

        try:
            with json_path.open("r", encoding="utf-8") as f:
                gold_data = json.load(f)
        except FileNotFoundError:
            continue

        rec = convert_xml_from_loaded(file_id=file_id, raw_text=raw_text, gold_data=gold_data)
        records.append(rec)

    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert gold JSON annotations + raw text to XML-tagged CSVs for SFT training."
    )
    parser.add_argument("--gold_dir", type=Path, required=True,
                        help="Directory containing gold JSON annotation files")
    parser.add_argument("--raw_text_csv", type=Path, required=True,
                        help="CSV with columns file_id, raw_text")
    parser.add_argument("--train_split", type=Path, required=True,
                        help="Text file listing train split filenames")
    parser.add_argument("--val_split", type=Path, required=True,
                        help="Text file listing val split filenames")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Output directory for train.csv and val.csv")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_text_df = pd.read_csv(args.raw_text_csv)

    train_ids = read_split_ids(args.train_split)
    val_ids = read_split_ids(args.val_split)

    train_df = process_split(train_ids, "train", args.gold_dir, raw_text_df)
    val_df = process_split(val_ids, "val", args.gold_dir, raw_text_df)

    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv", index=False)

    print(f"Saved: {args.out_dir / 'train.csv'} ({len(train_df)} rows)")
    print(f"Saved: {args.out_dir / 'val.csv'} ({len(val_df)} rows)")


if __name__ == "__main__":
    main()
