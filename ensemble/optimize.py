import optuna
import json
import ast
import pandas as pd
from pathlib import Path
from evaluate.evaluate import compute_span_f1, compute_token_f1, compute_shared_task_scores
from .mbr import load_predictions_for_one_or_all_docs, mbr_cluster_and_refine_single_doc

DATASET_DIR = Path(__file__).parent.parent / 'dataset'

# ========== CONFIGURATION ==========
PRED_LIST = [
    'model1_dft.json',
    'model2_dft_grpo.json',
    'model3_dft_unlabelp10.json'
]

GOLD_CSV   = DATASET_DIR / 'val.csv'
STATS_CSV  = DATASET_DIR / 'stats.csv'
SPLIT_FILE = DATASET_DIR / 'val.txt'

N_TRIALS = 5000 
# ===================================

def main():
    print("Loading predictions payload...")
    process_all, payload = load_predictions_for_one_or_all_docs(PRED_LIST, float('nan'))
    if not process_all:
        raise ValueError("Optimization currently expects multiple documents. Make sure sample_id is nan in mbr.py or PRED_LIST includes full jsons.")

    print("Loading gold annotations and raw text from CSV...")
    gold_annotations = {}
    raw_texts = {}
    
    if GOLD_CSV.exists():
        df = pd.read_csv(GOLD_CSV)
        for _, row in df.iterrows():
            f_id = str(row['file_id'])
            raw_text = str(row['raw_text'])
            offsets_str = str(row['gold_offsets'])
            
            raw_texts[f_id] = raw_text
            try:
                anns = ast.literal_eval(offsets_str)
                for ann in anns:
                    if 'category' in ann and isinstance(ann['category'], int):
                        ann['category'] = f"Category {ann['category']}"
                gold_annotations[f_id] = anns
            except Exception as e:
                print(f"Error parsing gold_offsets for {f_id}: {e}")
                gold_annotations[f_id] = []
    else:
        print(f"WARNING: GOLD_CSV '{GOLD_CSV}' not found. Evaluation won't work correctly.")
    
    if SPLIT_FILE and SPLIT_FILE.exists():
        split_ids = set()
        with open(SPLIT_FILE, 'r') as f:
            for line in f:
                split_ids.add(line.strip().removesuffix('.json'))
        
        gold_annotations = {k: v for k, v in gold_annotations.items() if k in split_ids}
        payload = {k: v for k, v in payload.items() if k in split_ids}
        print(f"Filtered to {len(gold_annotations)} gold files based on {SPLIT_FILE}")

    print(f"Loaded {len(payload)} documents for optimization, {len(gold_annotations)} gold items, {len(raw_texts)} raw texts.")

    # Print only new best score
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(trial):
        # You can reduce the search range if you are confident that the optimal value is around the current value
        # (but setting 0.0 - 1.0 covers the best value for 5000 trials)
        min_cluster_iou = trial.suggest_float("min_cluster_iou", 0.0, 1.0)
        min_pairwise_iou = trial.suggest_float("min_pairwise_iou", 0.0, 1.0)
        min_cluster_support = trial.suggest_float("min_cluster_support", 0.0, 1.0)
        min_iou_same_cat = trial.suggest_float("min_iou_same_cat", 0.0, 1.0)
        
        trial_predictions = {}
        for file_name, indexed_predictions in payload.items():
            result = mbr_cluster_and_refine_single_doc(
                indexed_predictions=indexed_predictions,
                min_cluster_iou=min_cluster_iou,
                min_pairwise_iou=min_pairwise_iou,
                min_cluster_support=min_cluster_support,
                min_iou_same_cat=min_iou_same_cat,
                verbose=False
            )
            trial_predictions[file_name] = result["final_predictions"]
            
        span_metrics = compute_span_f1(gold_annotations, trial_predictions, raw_texts)
        token_f1 = compute_token_f1(gold_annotations, trial_predictions, raw_texts)

        if STATS_CSV and STATS_CSV.exists():
            shared_task = compute_shared_task_scores(
                gold_annotations=gold_annotations,
                predictions=trial_predictions,
                raw_texts=raw_texts,
                stats_csv_path=STATS_CSV,
                overall_span_f1=span_metrics['f1'],
                overall_token_f1=token_f1,
            )
            score = shared_task['final_score']
        else:
            score = (span_metrics['f1'] + token_f1) / 2
            
        return score

    # SQLite to save progress
    study_name = "mbr_tuning_study"
    storage_name = "sqlite:///mbr_optuna.db"


    sampler = optuna.samplers.TPESampler()
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True, # If running again, load the previous value to continue
        sampler=sampler
    )

    # Force Optuna to test your default value as the first value!
    study.enqueue_trial({
        "min_cluster_iou": 0.1,
        "min_pairwise_iou": 0.1,
        "min_cluster_support": 0.34,
        "min_iou_same_cat": 0.8
    })

    print(f"Starting Optuna search for {N_TRIALS} trials... (Saving progress to {storage_name})")
    
    # Print only new best score
    def print_best_callback(study, trial):
        if study.best_trial.number == trial.number:
            print(f"🔥 New Best Score: {trial.value:.4f} at Trial {trial.number}")
            print(f"   Params: {trial.params}")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_best_callback])
    
    print("\n" + "="*50)
    print("Optimization finished!")
    print(f"Best score: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key} = {value:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()