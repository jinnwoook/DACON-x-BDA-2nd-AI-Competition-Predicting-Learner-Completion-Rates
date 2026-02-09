"""
SOTA Model 2: KoELECTRA-NSMC on train_detailed.csv
- 상세 텍스트 변환 데이터로 학습 (리더보드 최고 성능)
- Output: submission_model2_koelectra.csv, oof_model2_koelectra.csv
"""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    BASE_DIR = Path(__file__).parent.parent.parent.parent  # project root
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"

    TRAIN_DATA = "train_detailed.csv"
    TEST_DATA = "test_detailed.csv"

    MODEL_NAME = "monologg/koelectra-base-finetuned-nsmc"

    MAX_LEN = 256
    TRAIN_BS = 16
    EVAL_BS = 32
    EPOCHS = 8

    LR = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.05

    N_SPLITS = 5
    SEED = 42

    TARGET = "completed"
    ID_COL = "ID"


cfg = Config()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * focal_weight * ce_loss).mean()


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts.reset_index(drop=True) if hasattr(texts, 'reset_index') else texts
        self.labels = None if labels is None else np.asarray(labels).astype(int)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        if self.labels is None:
            return text
        return text, int(self.labels[idx])


def make_collate_fn(tokenizer, max_len):
    def collate(batch):
        if isinstance(batch[0], tuple):
            texts = [b[0] for b in batch]
            labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        else:
            texts = batch
            labels = None

        enc = tokenizer(texts, padding=True, truncation=True,
                       max_length=max_len, return_tensors="pt")
        if labels is not None:
            enc["labels"] = labels
        return enc
    return collate


def search_best_threshold(y_true, y_prob, pos_cap=0.70, step=0.005):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_thr, best_f1, best_pos = 0.5, -1.0, None

    for thr in np.arange(0.0, 1.0 + 1e-12, step):
        y_pred = (y_prob >= thr).astype(int)
        pos_rate = float(y_pred.mean())
        if pos_rate > pos_cap:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr, best_pos = f1, float(thr), pos_rate

    if best_f1 < 0:
        best_thr = 0.5

    return best_thr, best_f1, best_pos


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        if "labels" in batch:
            batch = {k: v for k, v in batch.items() if k != "labels"}
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, return_dict=True)
        p1 = torch.softmax(outputs.logits, dim=-1)[:, 1]
        probs.append(p1.cpu().numpy())
    return np.concatenate(probs)


def train_one_fold(fold, tr_idx, va_idx, texts, labels, test_texts, tokenizer, collate_fn):
    print(f"\n{'='*60}\nFold {fold}\n{'='*60}")

    tr_ds = TextDataset(texts.iloc[tr_idx], labels[tr_idx])
    va_ds = TextDataset(texts.iloc[va_idx], labels[va_idx])
    te_ds = TextDataset(test_texts, None)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.TRAIN_BS, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=cfg.EVAL_BS, shuffle=False, collate_fn=collate_fn)
    te_loader = DataLoader(te_ds, batch_size=cfg.EVAL_BS, shuffle=False, collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    loss_fn = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA,
                        label_smoothing=cfg.LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    total_steps = len(tr_loader) * cfg.EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_state, best_f1, best_thr = None, -1.0, 0.5
    patience, patience_counter = 3, 0

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(tr_loader), total=len(tr_loader),
                   desc=f"Fold {fold} Epoch {epoch}", leave=False)

        for step, batch in pbar:
            labels_t = batch["labels"].to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels_t)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            pbar.set_postfix({"loss": running_loss / (step + 1)})

        va_prob = predict_proba(model, va_loader, DEVICE)
        thr, f1c, _ = search_best_threshold(labels[va_idx], va_prob, pos_cap=0.70)

        print(f"[Fold {fold}] Epoch {epoch} | F1={f1c:.4f} (thr={thr:.3f})")

        if f1c > best_f1:
            best_f1, best_thr = f1c, thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    va_prob = predict_proba(model, va_loader, DEVICE)
    te_prob = predict_proba(model, te_loader, DEVICE)

    print(f"[Fold {fold}] Best F1={best_f1:.4f}, Threshold={best_thr:.3f}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return va_idx, va_prob, te_prob, best_thr, best_f1


def main():
    print("="*60)
    print("SOTA Model 2: KoELECTRA-NSMC on train_detailed.csv")
    print("="*60)

    train_df = pd.read_csv(cfg.DATA_DIR / cfg.TRAIN_DATA)
    test_df = pd.read_csv(cfg.DATA_DIR / cfg.TEST_DATA)

    print(f"\nTrain: {len(train_df)} rows, Test: {len(test_df)} rows")

    train_texts = train_df["text"].astype(str)
    y = train_df["label"].astype(int).values
    test_texts = test_df["text"].astype(str)
    test_ids = test_df["ID"].values

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    collate_fn = make_collate_fn(tokenizer, cfg.MAX_LEN)

    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)

    oof_prob = np.zeros(len(train_texts), dtype=float)
    test_prob_folds = []
    fold_info = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_texts, y), start=1):
        va_idx2, va_prob, te_prob, fold_thr, fold_f1 = train_one_fold(
            fold, tr_idx, va_idx, train_texts, y, test_texts, tokenizer, collate_fn
        )
        oof_prob[va_idx2] = va_prob
        test_prob_folds.append(te_prob)
        fold_info.append((fold, fold_thr, fold_f1))

    print(f"\nMean F1: {np.mean([s for _, _, s in fold_info]):.4f}")

    global_thr, global_f1, _ = search_best_threshold(y, oof_prob, pos_cap=0.70)
    test_prob_mean = np.mean(np.vstack(test_prob_folds), axis=0)
    test_pred = (test_prob_mean >= global_thr).astype(int)

    # Save OOF
    oof_df = pd.DataFrame({"ID": train_df["ID"], "prob": oof_prob, "label": y})
    oof_df.to_csv(cfg.OUTPUT_DIR / "oof_model2_koelectra.csv", index=False)

    # Save submission
    submission = pd.DataFrame({cfg.ID_COL: test_ids, cfg.TARGET: test_pred})
    submission.to_csv(cfg.OUTPUT_DIR / "submission_model2_koelectra.csv", index=False)

    # Save test probabilities
    prob_df = pd.DataFrame({"ID": test_ids, "prob": test_prob_mean})
    prob_df.to_csv(cfg.OUTPUT_DIR / "prob_model2_koelectra.csv", index=False)

    print(f"\nSaved: submission_model2_koelectra.csv")
    print(f"Global threshold: {global_thr:.4f}, F1: {global_f1:.4f}")


if __name__ == "__main__":
    main()
