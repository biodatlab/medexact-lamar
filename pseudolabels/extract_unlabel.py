"""Extract discharge summaries from NOTEEVENTS that are not in the MedDec splits."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Extract discharge summaries from MIMIC-III NOTEEVENTS "
        "excluding those already in MedDec train/val/test splits."
    )
    parser.add_argument(
        "--noteevents", required=True,
        help="Path to MIMIC-III NOTEEVENTS.csv",
    )
    parser.add_argument(
        "--split_files", required=True, nargs="+",
        help="Paths to split files (train.txt, val.txt, test.txt) to exclude",
    )
    parser.add_argument(
        "--output", default="other_discharge_summaries.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Load split file IDs to exclude
    exclude_files = set()
    for split_path in args.split_files:
        with open(split_path, "r") as f:
            for line in f:
                file_id = line.strip().split(".json")[0]
                if file_id:
                    exclude_files.add(file_id)
    print(f"Excluding {len(exclude_files)} files from splits.")

    # Load NOTEEVENTS and filter to discharge summaries
    print(f"Loading {args.noteevents} ...")
    summaries = pd.read_csv(args.noteevents)
    dx_sum = summaries[summaries["CATEGORY"] == "Discharge summary"].copy()
    print(f"Found {len(dx_sum)} discharge summaries.")

    # Build file_id and keep only needed columns
    dx_sum_clean = dx_sum[["SUBJECT_ID", "HADM_ID", "ROW_ID", "DESCRIPTION", "TEXT"]].copy()
    dx_sum_clean["file_id"] = dx_sum_clean.apply(
        lambda x: f"{int(x['SUBJECT_ID'])}_{int(x['HADM_ID'])}_{int(x['ROW_ID'])}",
        axis=1,
    )
    dx_sum_clean = dx_sum_clean[["file_id", "DESCRIPTION", "TEXT"]].copy()

    # Exclude files in splits
    dx_sum_clean = dx_sum_clean[~dx_sum_clean["file_id"].isin(exclude_files)].copy()
    dx_sum_clean = dx_sum_clean.reset_index(drop=True)
    print(f"Remaining after exclusion: {len(dx_sum_clean)} summaries.")

    dx_sum_clean.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
