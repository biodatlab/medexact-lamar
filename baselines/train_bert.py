import argparse
import ast
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from collections.abc import Iterable
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from pathlib import Path

# ================================================== #
# CLI-configurable args (override defaults from shell)
# ================================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='./roberta-base')
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--f1_mode', type=str, default='micro')
parser.add_argument('--ckpt_dir', type=str, default=None)
parser.add_argument('--last_ckpt_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--grad_accu', type=int, default=1)
parser.add_argument('--eval_only', action='store_true')
cli_args = parser.parse_args()

# derive checkpoint dirs from model_name + f1_mode + datetime if not explicitly set
model_basename = Path(cli_args.model_name).name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# ckpt_folder = f"./{model_basename}-{cli_args.f1_mode}"
ckpt_folder = f"./{model_basename}-{cli_args.f1_mode}--reassigned_{timestamp}"
if cli_args.ckpt_dir is None:
    cli_args.ckpt_dir = f"{ckpt_folder}/best_model.pt"
if cli_args.last_ckpt_dir is None:
    cli_args.last_ckpt_dir = f"{ckpt_folder}/final_model.pt"
# ================================================== #

args = Namespace(
    model_name      = cli_args.model_name,
    ckpt            = None,
    data_dir        = ".",
    label_encoding  = "multiclass",
    num_labels      = 19,
    num_decs        = 9,
    max_len         = cli_args.max_len,
    truncate_train  = False,
    truncate_eval   = False,
    seed            = [3407],
    lr              = 2e-5,
    weight_decay    = 0.001,
    batch_size      = cli_args.batch_size, #batchsize 32 for roberta
    grad_accumulation = cli_args.grad_accu,
    num_epoch       = 250,
    warmup_ratio    = 0.05,
    use_crf         = False,
    train_log       = 500,
    val_log         = 500,
    save_losses     = True,
    gpu             = 0,
    debug           = False,
    eval_only       = cli_args.eval_only,
    verbose         = True,
    ckpt_dir        = cli_args.ckpt_dir,
    last_ckpt_dir   = cli_args.last_ckpt_dir,
    f1_mode         = cli_args.f1_mode,
)

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

local_rank = setup_distributed()
device = f'cuda:{local_rank}'
all_losses = {'train': [], 'val': []}
valid_cats = range(0, 9)

def gen_splits(args):
    df = pd.read_csv(os.path.join(args.data_dir, 'gt_for_bert_reassigned.csv'))
    df['gold_offsets'] = df['gold_offsets'].apply(ast.literal_eval)
    train = df[df['split'] == 'train'].reset_index(drop=True)
    val   = df[df['split'] == 'val'].reset_index(drop=True)
    return train, val

def parse_cat(cat):
    for i, c in enumerate(cat):
        if c.isnumeric():
            if i + 1 < len(cat) and cat[i + 1].isnumeric():
                return int(cat[i:i + 2])
            return int(c)
    return None

class MyDataset(Dataset):
    def __init__(self, args, tokenizer, data_source, train=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data      = []
        self.train     = train

        for i, (_, row) in enumerate(data_source.iterrows()):
            self.data.append(self.load_decisions(args, row, i))

    def load_decisions(self, args, row, idx):
        text     = row['raw_text']
        annots   = row['gold_offsets']
        basename = str(row['file_id'])

        encoding = self.tokenizer(
            text,
            max_length = args.max_len,
            truncation = args.truncate_train if self.train else args.truncate_eval,
            padding    = 'max_length',
            return_offsets_mapping = False,
        )
        n_tokens = len(encoding['input_ids'])
        labels   = np.full(n_tokens, args.num_labels - 1, dtype=int)

        if not self.train:
            token_mask = np.ones(n_tokens)
            all_spans  = []

        # ── sort annotations by start then end offset ───────────────────────
        annots = sorted(annots, key=lambda a: (int(a['start_offset']), int(a['end_offset'])))

        for annot in annots:
            start     = int(annot['start_offset'])
            enc_start = encoding.char_to_token(start)
            for i in range(1, 10):
                if enc_start is not None:
                    break
                enc_start = encoding.char_to_token(start + i)
            if enc_start is None:
                break

            end     = int(annot['end_offset'])
            enc_end = encoding.char_to_token(end)
            for j in range(1, 10):
                if enc_end is not None:
                    break
                enc_end = encoding.char_to_token(end + j)
            if enc_end is None:
                enc_end = n_tokens

            if enc_end == enc_start:
                enc_end += 1

            cat = parse_cat(annot['category'])
            if cat is not None:
                cat -= 1
            if cat is None or cat not in valid_cats:
                if annot['category'] == 'TBD' and not self.train:
                    token_mask[enc_start:enc_end] = 0
                continue

            cat_b, cat_i = cat * 2, cat * 2 + 1
            if not any(x in [2 * y for y in range(args.num_labels // 2)]
                       for x in labels[enc_start:enc_end]):
                labels[enc_start] = cat_b
                if enc_end > enc_start + 1:
                    labels[enc_start + 1:enc_end] = cat_i

            if not self.train:
                all_spans.append({
                    'token_start': enc_start, 'token_end': enc_end - 1,
                    'label': cat, 'text_start': start, 'text_end': end,
                })

        result = {
            'input_ids': encoding['input_ids'],
            'labels':    labels,
            't2c':       encoding.token_to_chars,
        }
        if not self.train:
            result['all_spans']  = all_spans
            result['file_name']  = basename
            result['token_mask'] = token_mask
        return result

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)

def load_data(args):

    def collate_segment(batch):
        """Random-crop to max_len — sliding window augmentation for training."""
        xs, ys, masks, t2cs = [], [], [], []
        for item in batch:
            x, y = np.array(item['input_ids']), item['labels']
            n    = len(x)
            if n > args.max_len:
                start = np.random.randint(0, n - args.max_len + 1)
                x     = x[start:start + args.max_len]
                y     = y[start:start + args.max_len]
                mask  = [1] * args.max_len
            elif n < args.max_len:
                pad  = args.max_len - n
                x    = np.pad(x, (0, pad))
                y    = np.pad(y, (0, pad), constant_values=-100)
                mask = [1] * n + [0] * pad
            else:
                mask = [1] * n
            xs.append(x)
            ys.append(y)
            masks.append(mask)
            t2cs.append(item['t2c'])
        return {
            'input_ids': torch.tensor(np.array(xs)),
            'labels':    torch.tensor(np.array(ys)),
            'mask':      torch.tensor(masks),
            't2c':       t2cs,
        }

    def collate_full(batch):
        """Pad to longest in batch — preserves full sequence for eval."""
        lens    = [len(x['input_ids']) for x in batch]
        max_len = max(args.max_len, max(lens))
        for i, item in enumerate(batch):
            pad               = max_len - lens[i]
            item['input_ids'] = np.pad(item['input_ids'], (0, pad))
            item['labels']    = np.pad(item['labels'], (0, pad), constant_values=-100)
            item['mask']      = [1] * lens[i] + [0] * pad

        new_batch = {}
        for k in batch[0].keys():
            collated = [s[k] for s in batch]
            if k in ('all_spans', 'file_name', 't2c'):
                new_batch[k] = collated
            elif isinstance(batch[0][k], Iterable):
                new_batch[k] = torch.tensor(np.array(collated))
            else:
                new_batch[k] = collated
        return new_batch

    tokenizer       = load_tokenizer(args.model_name)
    args.vocab_size = tokenizer.vocab_size
    args.max_length = min(tokenizer.model_max_length, 512)

    train_df, val_df = gen_splits(args)

    train_dataset = MyDataset(args, tokenizer, train_df, train=True)
    val_dataset   = MyDataset(args, tokenizer, val_df)

    if is_main_process():
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_sampler    = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, collate_fn=collate_segment)
    train_ns         = DataLoader(train_dataset, 1,               shuffle=False, collate_fn=collate_full)
    val_dataloader   = DataLoader(val_dataset,   1,               shuffle=False, collate_fn=collate_full)

    return train_dataloader, val_dataloader, train_ns, train_sampler

class MyModel(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args       = args
        self.backbone   = backbone
        hidden_dim      = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, args.num_labels),
        )

    def forward(self, x, mask):
        """Forward pass through backbone + classifier (used by DDP)."""
        x    = x.to(self.backbone.device)
        mask = mask.to(self.backbone.device)
        out  = self.backbone(x, attention_mask=mask, output_attentions=False)
        return out, self.classifier(out.last_hidden_state)

    def decisions(self, x, mask):
        """Alias for forward — used in eval/generate paths."""
        return self.forward(x, mask)

    def generate(self, x, mask):
        """Non-overlapping sliding window inference over sequences of any length."""
        outs = []
        for offset in range(0, x.shape[1], self.args.max_len):
            segment      = x[:,    offset:offset + self.args.max_len]
            segment_mask = mask[:, offset:offset + self.args.max_len]
            h = self.decisions(segment, segment_mask)[0].last_hidden_state
            outs.append(h)
        return self.classifier(torch.cat(outs, dim=1))

def load_model(args, device):
    model = MyModel(args, AutoModel.from_pretrained(args.model_name, local_files_only=True)).to(device)

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
        if is_main_process():
            print(f"Loaded checkpoint: {args.ckpt}")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    crit         = nn.CrossEntropyLoss(reduction='none')
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer          = optimizer,
        num_warmup_steps   = int(args.warmup_ratio * args.total_steps),
        num_training_steps = args.total_steps,
    )
    return model, crit, optimizer, lr_scheduler

def indicators_to_spans(labels, idx=None):
    spans       = set()
    num_tokens  = len(labels)
    num_classes = args.num_labels // 2
    start, cat  = None, -1

    for t in range(num_tokens):
        prev_tag = labels[t - 1] if t > 0 else args.num_labels - 1
        cur_tag  = labels[t]

        if start is not None and cur_tag == cat + 1:
            continue
        elif start is not None:
            spans.add((idx, cat // 2, start, t - 1))
            start = None

        if start is None and (
            cur_tag in [2 * x for x in range(num_classes)]
            or (prev_tag == (args.num_labels - 1) and cur_tag != (args.num_labels - 1))
        ):
            start = t
            cat   = int(cur_tag) // 2 * 2

    return spans


def id_to_label(labels):
    result = []
    for l in labels:
        if   l == (args.num_labels - 1): result.append('O')
        elif l % 2 == 0:                 result.append('B-%d' % (l // 2))
        else:                            result.append('I-%d' % (l // 2))
    return result


def f1_score(ys, preds):
    tp = len(preds & ys)
    fp = len(preds) - tp
    fn = len(ys)   - tp
    return (2 * tp / (2 * tp + fp + fn)) * 100 if (tp + fp + fn) > 0 else 0.0


def calc_metrics_spans(ys, preds, span_ys=None):
    all_preds, all_ys = [], []
    for i, (y, pred) in enumerate(zip(ys, preds)):
        all_preds.append(indicators_to_spans(pred, idx=i))
        if span_ys is None:
            all_ys.append(indicators_to_spans(y.squeeze(), idx=i))

    all_preds = set().union(*all_preds)
    all_ys    = set(span_ys) if span_ys is not None else set().union(*all_ys)

    perclass = {
        c: f1_score(
            {x for x in all_ys    if x[1] == c},
            {x for x in all_preds if x[1] == c},
        )
        for c in range(args.num_decs)
    }

    if args.f1_mode == 'macro':
        f1 = np.mean(list(perclass.values())) if perclass else 0.0
    else:
        f1 = f1_score(all_ys, all_preds)

    return f1, all_preds, all_ys, perclass


def save_losses(model, crit, train_ns, val_dataloader):
    all_losses['train'].append(evaluate(model, train_ns,       crit, return_losses=True))
    all_losses['val'].append(  evaluate(model, val_dataloader, crit, return_losses=True))


def evaluate(model, dataloader, crit, return_losses=False, return_preds=False):
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    outs, ys, token_masks = [], [], []

    for batch in tqdm(dataloader, desc='Evaluating', leave=False,
                      disable=return_losses or not is_main_process()):
        x, y, mask = batch['input_ids'], batch['labels'], batch['mask']
        with torch.no_grad():
            logits = raw_model.generate(x, mask)
        outs.append(logits)
        ys.append(y)
        if 'token_mask' in batch:
            token_masks.append(batch['token_mask'])

    preds      = [x.squeeze() for x in outs]
    outs_stack = torch.cat([x.view(-1, args.num_labels) for x in outs], 0)
    ys_flat    = torch.cat([x.view(-1) for x in ys], 0).to(device)
    loss       = crit(outs_stack, ys_flat)
    preds      = [x.argmax(-1) for x in preds]
    preds_stack= outs_stack.argmax(-1)

    if return_losses:
        lens, losses, offset = [x.shape[0] for x in outs], [], 0
        for ln in lens:
            losses.append(loss[offset:offset + ln].mean().item())
            offset += ln
        return losses

    loss = loss.mean()

    if token_masks:
        token_masks = torch.cat(token_masks, 1).squeeze().to(device)
        acc = ((ys_flat == preds_stack).float() * token_masks).sum() / token_masks.sum() * 100
    else:
        acc = (ys_flat == preds_stack).float().mean() * 100

    if 'all_spans' in dataloader.dataset.data[0]:
        all_spans = [x['all_spans'] for x in dataloader.dataset.data]
        span_ys   = [
            (i, s['label'], s['token_start'], s['token_end'])
            for i, spans in enumerate(all_spans)
            for s in spans
        ]
    else:
        span_ys = None

    f1, span_preds, span_ys, perclass = calc_metrics_spans(ys, preds, span_ys)

    if return_preds:
        return span_preds, span_ys

    raw_model.train()
    return {'f1': f1, 'acc': acc}, loss, perclass

def train(args, model, crit, optimizer, lr_scheduler,
          train_dataloader, val_dataloader, verbose=True, train_ns=None,
          train_sampler=None):

    step, best_f1, best_acc, best_step = 0, -1, 0, 0
    best_perclass = None
    epoch = 0
    train_sampler.set_epoch(epoch)
    train_iter    = iter(train_dataloader)
    losses        = []
    pbar = tqdm(total=args.total_steps, desc='Training', dynamic_ncols=True, position=0, leave=True,
                disable=not is_main_process())
    while step < args.total_steps:
        batch = next(train_iter, None)
        if batch is None:
            epoch += 1
            train_sampler.set_epoch(epoch)
            train_iter = iter(train_dataloader)
            continue


        x    = batch['input_ids'].to(device)
        mask = batch['mask'].to(device)
        y    = batch['labels'].to(device)

        _, logits = model(x, mask)
        loss      = crit(logits.view(-1, args.num_labels), y.view(-1)).mean()


        losses.append(loss.item())
        is_update_step = (step + 1) % args.grad_accumulation == 0

        # If not update step, tell DDP not to sync gradients across GPUs
        if not is_update_step:
            with model.no_sync():
                (loss / args.grad_accumulation).backward()
        else:
            (loss / args.grad_accumulation).backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        if is_main_process() and step % (args.train_log * args.grad_accumulation) == 0:
            pbar.set_postfix({'loss': f'{np.mean(losses):.4f}', 'best_f1': f'{best_f1:.2f}'})
            losses = []

        if len(val_dataloader) > 0 and step % (args.val_log * args.grad_accumulation) == 0 and step > 0:
            if args.save_losses:
                save_losses(model, crit, train_ns, val_dataloader)

            metrics_out, val_loss, perclass = evaluate(model, val_dataloader, crit)
            f1, acc = metrics_out['f1'], metrics_out['acc']
            if verbose and is_main_process():
                tqdm.write(f"[val] step {step:5d} | f1: {f1:.2f} | acc: {acc:.2f} | loss: {val_loss:.4f}")

            # if f1 > best_f1:
            #     best_f1, best_acc, best_step = f1, acc, step
            #     best_perclass = perclass
            #     if not args.debug and is_main_process():
            #         os.makedirs(os.path.dirname(args.ckpt_dir), exist_ok=True)
            #         torch.save(model.module.state_dict(), args.ckpt_dir)
            #         print(f"  ✓ best model saved (f1={best_f1:.2f})")
            #     dist.barrier()

            if f1 > best_f1:
                best_f1, best_acc, best_step = f1, acc, step
                best_perclass = perclass
                if not args.debug and is_main_process():
                    os.makedirs(os.path.dirname(args.ckpt_dir), exist_ok=True)
                    torch.save(model.module.state_dict(), args.ckpt_dir)
                    print(f"  ✓ best model saved (f1={best_f1:.2f})")

            if not args.debug:
                dist.barrier()

        if not args.debug and step == args.total_steps - 1:
            if is_main_process():
                os.makedirs(os.path.dirname(args.last_ckpt_dir), exist_ok=True)
                torch.save(model.module.state_dict(), args.last_ckpt_dir)
                print(f"  ✓ last model saved")
            dist.barrier()
        pbar.update(1)
        step += 1

    if is_main_process():
        print(f"\nBest → step {best_step} | f1: {best_f1:.2f} | acc: {best_acc:.2f}")
        if best_perclass:
            for c, f in best_perclass.items():
                print(f"  class {c:2d}: {f:.2f}")

    return best_f1, best_acc, best_step

def indicators_to_char_spans(labels, t2c):
    """Convert predicted token labels → character-offset spans using token_to_chars."""
    token_spans = indicators_to_spans(labels)
    char_spans  = []
    for _, cat, tok_start, tok_end in sorted(token_spans, key=lambda x: x[2]):
        start_chars = t2c(tok_start)
        end_chars   = t2c(tok_end)
        if start_chars is None or end_chars is None:
            continue
        char_spans.append({
            'category':     'Category %d' % (cat + 1),
            'start_offset': start_chars.start,
            'end_offset':   end_chars.end,
        })
    return char_spans


def predict_dataframe(model, tokenizer, df, args):
    """Run inference on a raw DataFrame with a 'raw_text' column.
    Returns the DataFrame with a new 'predictions' column."""
    model.eval()
    all_preds = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Predicting'):
        text     = row['raw_text']
        encoding = tokenizer.encode_plus(text)
        x        = torch.tensor(encoding['input_ids']).unsqueeze(0).to(device)
        mask     = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(device)

        with torch.no_grad():
            out  = model.generate(x, mask)
        pred = out.argmax(-1).squeeze().cpu().numpy()

        char_spans = indicators_to_char_spans(pred, encoding.token_to_chars)
        # attach the matched text for readability
        for s in char_spans:
            s['text'] = text[s['start_offset']:s['end_offset']]
        all_preds.append(char_spans)

    df = df.copy()
    df['predictions'] = all_preds
    return df

def main(args):
    f1s = []
    for seed in args.seed:
        if is_main_process():
            print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.seed_val = seed

        train_dataloader, val_dataloader, train_ns, train_sampler = load_data(args)
        args.total_steps = args.num_epoch * len(train_dataloader)
        if is_main_process():
            print(f"Total steps: {args.total_steps} "
                  f"({args.num_epoch} epochs × {len(train_dataloader)} batches)")

        model, crit, optimizer, lr_scheduler = load_model(args, device)

        if not args.eval_only:
            f1, acc, step = train(
                args, model, crit, optimizer, lr_scheduler,
                train_dataloader, val_dataloader, args.verbose, train_ns,
                train_sampler,
            )
            f1s.append(f1)
            if is_main_process():
                print(f"\nFinal → seed {seed} | F1: {f1:.2f} | Acc: {acc:.2f}")

        else:
            metrics_out, loss, perclass = evaluate(model, val_dataloader, crit)
            f1, acc = metrics_out['f1'], metrics_out['acc']
            if is_main_process():
                print(f"[Val] F1: {f1:.2f} | Acc: {acc:.2f} | Loss: {loss:.4f}")
            f1s.append(f1)

        if args.save_losses and is_main_process():
            np.savez(f'./{Path(args.ckpt_dir).parent}/losses_{seed}.npz',
                     train=all_losses['train'], val=all_losses['val'])

    if is_main_process():
        print(f"\nMean F1 across seeds: {np.mean(f1s):.2f}")

    cleanup_distributed()
    return np.mean(f1s)

main(args)
