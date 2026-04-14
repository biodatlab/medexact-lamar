import pandas as pd
import numpy as np
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATASET_DIR = Path(__file__).parent.parent / 'dataset'
EMB_DIR = Path(__file__).parent.parent / 'embeddings'

DATASETS = [
    {
        "name": "other_discharge",
        "csv": DATASET_DIR / "other_discharge_summaries.csv",
        "text_col": "TEXT",
        "filter": lambda df: df[df['DESCRIPTION'] == 'Report'].reset_index(drop=True),
        "emb_dir": EMB_DIR / "other_discharge",
        "out_emb": EMB_DIR / "report_embeddings.npy",
        "out_meta": EMB_DIR / "report_metadata.csv",
    },
    {
        "name": "val",
        "csv": DATASET_DIR / "val.csv",
        "text_col": "raw_text",
        "filter": None,
        "emb_dir": EMB_DIR / "val",
        "out_emb": EMB_DIR / "val_embeddings.npy",
        "out_meta": EMB_DIR / "val_metadata.csv",
    },
    {
        "name": "test",
        "csv": DATASET_DIR / "test.csv",
        "text_col": "raw_text",
        "filter": None,
        "emb_dir": EMB_DIR / "test",
        "out_emb": EMB_DIR / "test_embeddings.npy",
        "out_meta": EMB_DIR / "test_metadata.csv",
    },
    {
        "name": "train",
        "csv": DATASET_DIR / "train.csv",
        "text_col": "raw_text",
        "filter": None,
        "emb_dir": EMB_DIR / "train",
        "out_emb": EMB_DIR / "train_embeddings.npy",
        "out_meta": EMB_DIR / "train_metadata.csv",
    },
]

BATCH_SIZE = 8


# Load model once
print("Loading model...")
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B",
    trust_remote_code=True,
    model_kwargs={
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
    },
    tokenizer_kwargs={"padding_side": "left"},
)

model[0].auto_model = torch.compile(model[0].auto_model, mode="max-autotune")
_ = model.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
torch.cuda.empty_cache()
print("torch.compile warmup done\n")


# Encode + save + merge function
def encode_dataset(cfg):
    print(f"\n{'='*60}")
    print(f"Dataset: {cfg['name']}")
    print(f"{'='*60}")

    df = pd.read_csv(cfg["csv"])
    if cfg["filter"] is not None:
        df = cfg["filter"](df)
    texts = df[cfg["text_col"]].tolist()
    print(f"Total rows: {len(df)}")

    cfg["emb_dir"].mkdir(parents=True, exist_ok=True)

    already_done = {
        int(f.stem.replace('emb_', ''))
        for f in cfg["emb_dir"].glob('emb_*.npy')
    }
    print(f"Already done: {len(already_done)}/{len(texts)}")
    print(f"Remaining: {len(texts) - len(already_done)}\n")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Encoding {cfg['name']}"):
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(texts))))

        if all(idx in already_done for idx in batch_indices):
            continue

        batch_texts = [texts[idx] for idx in batch_indices]
        embs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

        for j, idx in enumerate(batch_indices):
            if idx not in already_done:
                np.save(f'{cfg["emb_dir"]}/emb_{idx}.npy', embs[j])

        if (i // BATCH_SIZE) % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nEncoding complete! Merging...")
    all_embs = [np.load(f'{cfg["emb_dir"]}/emb_{idx}.npy') for idx in tqdm(range(len(texts)), desc="Merging")]
    embeddings = np.stack(all_embs, axis=0)
    print(f"Shape: {embeddings.shape} | Size: {embeddings.nbytes / 1e6:.0f} MB")

    np.save(cfg["out_emb"], embeddings)
    df.to_csv(cfg["out_meta"], index=False)
    print(f"Saved: {cfg['out_emb']}")
    print(f"Saved: {cfg['out_meta']}")



# Run all datasets
for cfg in DATASETS:
    encode_dataset(cfg)

print("\nAll datasets encoded successfully!")
