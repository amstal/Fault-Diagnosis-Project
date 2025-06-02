#!/usr/bin/env python
"""Train_Dann.py – Domain‑Adversarial Neural Network (DANN)
===========================================================
* Entraîne un DANN pour le diagnostic de défauts à partir d'un jeu simulation
  (source) et d'un jeu réel (cible).
* Sauvegarde **poids du modèle** + **indices exacts** du split cible
  (unsup / test) afin de garantir une évaluation identique par la suite.

"""
import argparse, random, numpy as np, torch, os
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import StratifiedShuffleSplit

from models import DANN, load_csv_from_dir, categories


def get_args():
    p = argparse.ArgumentParser(description="Train DANN model for fault diagnosis")
    p.add_argument("--simu-dir", default="./trainingDatasets/20241016")
    p.add_argument("--real-dir", default="./testDatasets/20241016")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--target-train-ratio", type=float, default=0.5)
    p.add_argument("--no-stratified", action="store_true")
    p.add_argument("--noise-std", type=float, default=0)
    p.add_argument("--output", default=None)
    return p.parse_args()

args = get_args()


SIMU_DIR = args.simu_dir
REAL_DIR = args.real_dir
EPOCHS   = args.epochs
BATCH    = args.batch_size
LR       = args.lr
SEED     = args.seed
SRC_RATIO= args.train_ratio
TGT_RATIO= args.target_train_ratio
STRAT    = not args.no_stratified
NOISE_STD= args.noise_std

WEIGHT_DECAY = 1e-5
MOMENTUM     = 0.9
T0, Tmult, ETA_MIN = 40, 2, 1e-6
NOISE_MEAN, NOISE_PROB = 0.0, 0.8
LABEL_SMOOTH = 0.05

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = (torch.device("cuda") if torch.cuda.is_available() else
          torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cpu"))
print(f"Training on: {device}")



def stratified_split(X: torch.Tensor, y: torch.Tensor, train_ratio: float):
    """Return X_tr, y_tr, X_val, y_val, idx_tr, idx_val."""
    if STRAT:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio,
                                     random_state=SEED)
        idx_tr, idx_val = next(sss.split(torch.zeros(len(y)), y.cpu()))
    else:
        g = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(len(y), generator=g)
        cut = int(train_ratio * len(y))
        idx_tr, idx_val = perm[:cut], perm[cut:]
    idx_tr = torch.as_tensor(idx_tr, device=X.device)
    idx_val= torch.as_tensor(idx_val, device=X.device)
    return (X[idx_tr], y[idx_tr], X[idx_val], y[idx_val], idx_tr, idx_val)


def augment_source(x: torch.Tensor):
    if NOISE_STD == 0 or NOISE_PROB == 0:
        return x
    mask = (torch.rand(len(x), device=x.device) < NOISE_PROB).unsqueeze(-1).unsqueeze(-1)
    noise = torch.randn_like(x) * NOISE_STD + NOISE_MEAN
    return torch.where(mask, x + noise, x)


def nll_loss_ls(logits, target):
    logp = F.log_softmax(logits, dim=1)
    true_dist = torch.zeros_like(logp)
    true_dist.fill_(LABEL_SMOOTH / (logits.size(1) - 1))
    true_dist.scatter_(1, target.unsqueeze(1), 1.0 - LABEL_SMOOTH)
    return torch.mean(torch.sum(-true_dist * logp, dim=1))


print("→ Loading simulation (source)…")
Xs, ys, mean, std = load_csv_from_dir(SIMU_DIR, device=device)
print(f"  Simulation set: {len(Xs)} samples")

print("→ Loading real (target)…")
Xt_full, yt_full, _, _ = load_csv_from_dir(REAL_DIR, mean, std, device=device)
print(f"  Real set     : {len(Xt_full)} samples (before split)")

Xt_unsup, y_unsup, Xtest, ytest, idx_unsup, idx_test = \
    stratified_split(Xt_full, yt_full, TGT_RATIO)
print(f"  Target split : unsup={len(Xt_unsup)} | test={len(Xtest)}")
Xt = Xt_unsup

split_filename = f"target_split_seed{SEED}.npz"
np.savez(split_filename,
         idx_unsup=idx_unsup.cpu().numpy(),
         idx_test =idx_test.cpu().numpy())
print(f"  → Indices saved to {split_filename}")


print("→ Preparing simulation splits…")
X_train, y_train, X_val, y_val, _, _ = stratified_split(Xs, ys, SRC_RATIO)
print(f"  Simulation split – train={len(X_train)} | val={len(X_val)}")


model = DANN(in_channels=6, num_classes=len(categories)).to(device)
optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=T0, T_mult=Tmult,
                                        eta_min=ETA_MIN)


history = {k: [] for k in ["loss_cls", "loss_dom", "val_acc", "test_acc", "lr"]}
print("\n=== Training =================================================================")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_cls = tot_dom_s = tot_dom_t = correct = 0

    p = epoch / EPOCHS
    alpha = 2.0 / (1 + np.exp(-10 * p)) - 1.0
    lambda_dom = 0.1 * (1 - p)

    for i in range(0, len(X_train), BATCH):
        xs = augment_source(X_train[i:i+BATCH])
        ys_b = y_train[i:i+BATCH]

        t_start = i % len(Xt)
        xt_b = Xt[t_start:t_start+len(xs)]
        if len(xt_b) < len(xs):
            xt_b = torch.cat([xt_b, Xt[:len(xs)-len(xt_b)]], dim=0)

        dom_s_lbl = torch.zeros(len(xs), device=device, dtype=torch.long)
        dom_t_lbl = torch.ones (len(xt_b), device=device, dtype=torch.long)

        optimiser.zero_grad()
        cls_logits, dom_logits_s = model(xs, alpha)
        loss_cls = nll_loss_ls(cls_logits, ys_b)
        loss_dom_s = F.cross_entropy(dom_logits_s, dom_s_lbl)
        _, dom_logits_t = model(xt_b, alpha)
        loss_dom_t = F.cross_entropy(dom_logits_t, dom_t_lbl)
        loss = loss_cls + lambda_dom * (loss_dom_s + loss_dom_t)
        loss.backward(); optimiser.step()

        batch = len(xs)
        tot_cls   += loss_cls.item() * batch
        tot_dom_s += loss_dom_s.item() * batch
        tot_dom_t += loss_dom_t.item() * batch
        correct   += (torch.argmax(cls_logits, 1) == ys_b).sum().item()

    model.eval()
    with torch.no_grad():
        val_logits, _  = model(X_val)
        test_logits, _ = model(Xtest)
    val_acc  = accuracy_score(y_val.cpu(),  torch.argmax(val_logits, 1).cpu())
    test_acc = accuracy_score(ytest.cpu(), torch.argmax(test_logits, 1).cpu())

    scheduler.step()
    lr_cur = scheduler.get_last_lr()[0]

    n_train = len(X_train)
    history["loss_cls"].append(tot_cls / n_train)
    history["loss_dom"].append((tot_dom_s + tot_dom_t) / n_train)
    history["val_acc" ].append(val_acc)
    history["test_acc"].append(test_acc)
    history["lr"      ].append(lr_cur)

    train_acc = correct / n_train
    print(
        f"Epoch {epoch:3d}/{EPOCHS} | "
        f"loss cls={history['loss_cls'][-1]:.4f} dom={history['loss_dom'][-1]:.4f} | "
        f"acc train={train_acc*100:5.1f}% val={val_acc*100:5.1f}% test={test_acc*100:5.1f}% | "
        f"lr={lr_cur:.2e}")


print("\nFinal validation / test accuracies:")
print(f"  Val  : {history['val_acc'][-1]*100:.2f}%")
print(f"  Test : {history['test_acc'][-1]*100:.2f}%")

if args.output:
    torch.save(model.state_dict(), args.output)
    print(f"Model weights saved to {args.output}")
print("✓ Training complete.")
