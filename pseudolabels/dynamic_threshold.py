"""Selection of threshold, Pseudolable filtering, and mixing selection with train set"""

from email import parser

import pandas as pd
import os
import json
from pathlib import Path
import argparse
import tqdm
import sys
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluate.evaluate import compute_span_f1, compute_token_f1, load_gold_annotations
from utils.pred_to_offset import extract_spans_from_tags, find_span_in_raw

def parser_args():
    parser = argparse.ArgumentParser(
        description="Psedolabel data exploration and selection"
    )
    parser.add_argument(
        "--pred_other_csv", required=True,
        help="Path to inferenced discharge summaries outside of MedDec",
    )
    # ------------------- Threshold exploration -------------------
    parser.add_argument(
        "--pred_val_csv",
        help="Paths to inferenced validation set with entropy for threshold exploration",
    )
    parser.add_argument("--gold_dir", 
                    help="Directory containing gold JSON annotation files"),
    parser.add_argument(("--threshold_list"), nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25,0.30, 0.5, 0.75],
                        help="List of entropy thresholds to evaluate for pseudolabel selection")
    # ------------------- Train + Psedolabel mixing -------------------
    parser.add_argument(
        "--train_csv",
        help="Path to tag-converted train.csv for data mixing ",
    )
    parser.add_argument(
        "--threshold", default=None, type=float,
        help="Quantile for pseudolabel filtering (e.g. 0.10 for 10th percentile)"
    )
    parser.add_argument(
        "--output_csv", default=None,
        help="Output of the mixed data"
    )
    args = parser.parse_args()
    if (args.pred_val_csv is None) != (args.gold_dir is None):
        parser.error("--pred_val, --gold_dir, and --threshold_list must be provided together.")
    if (args.train_csv is None) != (args.threshold is None):
        parser.error("--train_csv and --threshold must be provided together.")
    if args.output_csv is None:
        args.output_csv = f"train_with_q{args.threshold}.csv"
    return args

def plot_entropy_vs_scores(df, name, save_path=None, dpi=300):
    df = df.copy()
    df["base_score"] = (df["token_f1"] + df["span_f1"]) / 2

    overall_span_rho, overall_span_p = spearmanr(df["entropy"], df["span_f1"])
    overall_token_rho, overall_token_p = spearmanr(df["entropy"], df["token_f1"])
    overall_base_rho, overall_base_p = spearmanr(df["entropy"], df["base_score"])

    print(f"\n{name} Overall:")
    print("span_f1     rho=", overall_span_rho, "p=", overall_span_p)
    print("token_f1    rho=", overall_token_rho, "p=", overall_token_p)
    print("base_score  rho=", overall_base_rho, "p=", overall_base_p)

    plt.figure(figsize=(10, 6))

    # keep all score dots
    plt.scatter(df["entropy"], df["span_f1"], alpha=0.4, label="span_f1")
    plt.scatter(df["entropy"], df["token_f1"], alpha=0.4, label="token_f1")
    plt.scatter(df["entropy"], df["base_score"], alpha=0.4, label="base_score")

    # only base_score trend line
    x = df["entropy"]
    y = df["base_score"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    plt.plot(x_sorted, p(x_sorted), linewidth=2, label="base_score trend")

    # percentile cutoffs: p10 red, others grey
    percentiles = [10, 15, 20, 25]
    for pct in percentiles:
        cutoff = np.percentile(df["entropy"], pct)
        color = "red" if pct == 10 else "grey"
        plt.axvline(
            cutoff,
            linestyle="--",
            alpha=0.8,
            color=color,
            label=f"p{pct} cutoff"
        )

    plt.xlabel("Entropy")
    plt.ylabel("Score")
    plt.title(
        f"{name}: Entropy vs scores\n"
        f"Overall Spearman: span={overall_span_rho:.3f}, "
        f"token={overall_token_rho:.3f}, "
        f"base={overall_base_rho:.3f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close()

def threshold_exploration(pred_val, threshold_list, gold_dir, pred_other_csv):
    gold_dir = Path(gold_dir)
    val_df = pd.read_csv(pred_val)
    other_df = pd.read_csv(pred_other_csv)
    span_f1_list = []
    token_f1_list = []
    for idx, row in tqdm.tqdm(val_df.iterrows(), total=len(val_df)):
        file_id = str(row['file_id'])
        raw_text = str(row['raw_text'])
        tag_pred = str(row['predicted'])

        # Extract predicted spans from tagged text and align to raw text offsets
        pred_spans = extract_spans_from_tags(tag_pred)
        aligned = []
        for span in pred_spans:
            result = find_span_in_raw(span['text'], raw_text, expected_ratio=span['pos_ratio'])
            if result:
                start, end, _ = result
                aligned.append({
                    'start_offset': start,
                    'end_offset': end,
                    'category': span['category'],
                })
        aligned.sort(key=lambda x: (x['start_offset'], x['end_offset']))

        gold_file = gold_dir / f'{file_id}.json'
        if gold_file.exists():
            gold_anns = load_gold_annotations(str(gold_file))
        else:
            gold_anns = {file_id: []}

        # Build single-file dicts matching the expected signatures
        preds = {file_id: aligned}
        raw_texts = {file_id: raw_text}

        # Compute metrics
        span_result = compute_span_f1(gold_anns, preds, raw_texts)
        token_result = compute_token_f1(gold_anns, preds, raw_texts)

        span_f1_list.append(span_result['f1'])
        token_f1_list.append(token_result)

    val_df['span_f1'] = span_f1_list
    val_df['token_f1'] = token_f1_list

    thresholds = val_df['entropy'].quantile(threshold_list).to_dict()
    rows = []
    for q, thr in thresholds.items():
        kept = val_df[val_df['entropy'] <= thr]
        rows.append({
            'quantile': q,
            'threshold': thr,
            'kept_n': len(kept),
            'kept_frac': len(kept) / len(val_df),
            'mean_span_f1': kept['span_f1'].mean(),
            'mean_token_f1': kept['token_f1'].mean(),
            'base_score': (kept['span_f1'].mean()+kept['token_f1'].mean())/2
        })
    
    df = pd.DataFrame(rows)
    print("="*30, "Validation Set Threshold Exploration", "="*30)
    print(df.to_string(
        formatters={
            'kept_frac': '{:.4f}'.format,
            'mean_span_f1': '{:.4f}'.format,
            'mean_token_f1': '{:.4f}'.format,
            'base_score': '{:.4f}'.format,
        }
    ))
    print('\n',"="*30, "Unlabel Set Threshold Exploration", "="*30)
    other_kept = pd.DataFrame([
    {
        'quantile': q,
        'threshold': thr,
        'kept_frac': len(kept) / len(val_df),
        'kept_n': (other_df['entropy'] <= thr).sum(),
    }
    for q, thr in other_df['entropy'].quantile(threshold_list).to_dict().items()
    ])
    print(other_kept.to_string(formatters={'kept_frac': '{:.4f}'.format}))
    fig_save_path = f"{Path(pred_val).stem}_threshold_exploration.png"
    plot_entropy_vs_scores(val_df, Path(pred_val).stem, save_path=fig_save_path)
    print(f"\nSaved entropy vs score plot to {fig_save_path}")

def main():
    args = parser_args()
    if args.pred_val_csv and args.gold_dir:
        threshold_exploration(args.pred_val_csv, args.threshold_list, args.gold_dir, args.pred_other_csv)
    
    if args.train_csv and args.threshold:
        train_df = pd.read_csv(args.train_csv)
        other_df = pd.read_csv(args.pred_other_csv)
        threshold_value = other_df['entropy'].quantile(args.threshold)
        selected_pseudo = other_df[other_df['entropy'] <= threshold_value]
        print(f"Selected {len(selected_pseudo)} pseudolabels with entropy <= {threshold_value:.4f} (q{args.threshold})")
        mixed_df = pd.concat([train_df, selected_pseudo], ignore_index=True)
        mixed_df.to_csv(args.output_csv, index=False)
        print(f"Saved mixed data to {args.output_csv}")

if __name__ == "__main__":
    main()