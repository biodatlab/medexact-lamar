import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

EMB_DIR = Path(__file__).parent.parent / 'embeddings'
OUT_DIR = Path(__file__).parent.parent / 'retrieval'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
CANDIDATES = 100
BATCH_SIZE = 128

# Load embeddings
train_emb = np.load(EMB_DIR / 'train_embeddings.npy')
train_df  = pd.read_csv(EMB_DIR / 'train_metadata.csv')

val_emb = np.load(EMB_DIR / 'val_embeddings.npy')
val_df  = pd.read_csv(EMB_DIR / 'val_metadata.csv')

test_emb = np.load(EMB_DIR / 'test_embeddings.npy')
test_df  = pd.read_csv(EMB_DIR / 'test_metadata.csv')

ref_emb = np.load(EMB_DIR / 'report_embeddings.npy')
ref_df  = pd.read_csv(EMB_DIR / 'report_metadata.csv')

print(f"Train: {train_emb.shape}")
print(f"Val:   {val_emb.shape}")
print(f"Test:  {test_emb.shape}")
print(f"Ref:   {ref_emb.shape}")
print(f"Total queries: {len(train_emb) + len(val_emb) + len(test_emb)}")

# Combine train + val + test
all_emb = np.concatenate([train_emb, val_emb, test_emb], axis=0)
all_df = pd.concat([
    train_df.assign(split='train'),
    val_df.assign(split='val'),
    test_df.assign(split='test'),
], ignore_index=True)

print(f"Combined queries: {all_emb.shape}")

# Batch similarity
all_top_indices = []
all_top_scores = []

for i in tqdm(range(0, len(all_emb), BATCH_SIZE), desc="Computing similarities"):
    batch = all_emb[i : i + BATCH_SIZE]
    scores = batch @ ref_emb.T

    top_idx = np.argsort(scores, axis=1)[:, -CANDIDATES:][:, ::-1]
    top_sc = np.take_along_axis(scores, top_idx, axis=1)

    all_top_indices.append(top_idx)
    all_top_scores.append(top_sc)

top_indices = np.concatenate(all_top_indices, axis=0)
top_scores = np.concatenate(all_top_scores, axis=0)

# Greedy assignment (no duplicate refs across splits)
print("\nGreedy assigning...")

pairs = [
    (float(top_scores[q, c]), q, int(top_indices[q, c]))
    for q in range(len(all_emb))
    for c in range(CANDIDATES)
]
pairs.sort(key=lambda x: x[0], reverse=True)

claimed_refs = set()
query_assignments = {}

for score, q_idx, r_idx in tqdm(pairs, desc="Assigning pairs"):
    if r_idx in claimed_refs:
        continue
    if q_idx in query_assignments and len(query_assignments[q_idx]) >= TOP_K:
        continue

    query_assignments.setdefault(q_idx, []).append((score, r_idx))
    claimed_refs.add(r_idx)

print(f"Total unique refs used: {len(claimed_refs)}")
incomplete = sum(1 for v in query_assignments.values() if len(v) < TOP_K)
if incomplete:
    print(f"Warning: {incomplete} queries have fewer than {TOP_K} matches.")

# Build result DataFrame
rows = []
for q_idx in tqdm(range(len(all_df)), desc="Building results"):
    assignments = sorted(query_assignments.get(q_idx, []), key=lambda x: x[0], reverse=True)
    for rank, (score, r_idx) in enumerate(assignments, 1):
        rows.append({
            'split':          all_df.iloc[q_idx]['split'],
            'query_idx':      q_idx,
            'query_file_id':  all_df.iloc[q_idx].get('file_id', q_idx),
            'query_raw_text': all_df.iloc[q_idx]['raw_text'],
            'rank':           rank,
            'score':          round(score, 4),
            'ref_idx':        r_idx,
            'ref_file_id':    ref_df.iloc[r_idx]['file_id'],
            'ref_text':       ref_df.iloc[r_idx]['TEXT'],
        })

result_df = pd.DataFrame(rows)

# Save
for split in ['train', 'val', 'test']:
    subset = result_df[result_df['split'] == split].drop(columns='split')
    subset.to_csv(OUT_DIR / f'top{TOP_K}_matches_{split}.csv', index=False)
    print(f"Saved {split}: {len(subset)} rows")

result_df.to_csv(OUT_DIR / f'top{TOP_K}_matches_all.csv', index=False)

dup_check = result_df['ref_idx'].duplicated().sum()
print(f"Global duplicate refs: {dup_check} (should be 0)")
