"""Training for one fold, and prediction for one song."""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml import config
from ml.dataset import ChunkDataset, FrameDataset, compute_norm
from ml.models import build_model


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#prediction
@torch.no_grad()
def predict_probs(model, name, song, mean, std, context=config.CONTEXT, device="cpu", batch=1024):
    """25 x N class probabilities for one song."""
    model.eval()
    C = ((song["cqt"][0] - mean) / std).astype(np.float32)
    if name == "crnn":# CRNN takes the whole song at once
        return torch.softmax(model(torch.from_numpy(C)[None, None].to(device)), dim=1)[0].cpu().numpy()
    Cp = np.pad(C, ((0, 0), (context, context)))
    n = C.shape[1]
    out = []
    for s in range(0, n, batch):
        x = np.stack([Cp[:, j:j + 2 * context + 1] for j in range(s, min(s + batch, n))])[:, None]
        out.append(torch.softmax(model(torch.from_numpy(x).to(device)), dim=1).cpu().numpy())
    return np.concatenate(out).T


def frame_accuracy(model, name, songs, mean, std, context, device):
    correct = total = 0
    for s in songs:
        pred = predict_probs(model, name, s, mean, std, context, device).argmax(0)
        correct += int((pred == s["classes"]).sum())
        total += len(pred)
    return correct / max(total, 1)


#training
def train_model(name, songs, context=config.CONTEXT, width=None, augment="none", shifts=None,
                epochs=10, lr=1e-3, batch_size=256, device=None, verbose=True):
    """Train on the songs, keep 10% aside to pick the best epoch. Returns model, mean, std, history."""
    set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    songs = list(songs)
    random.shuffle(songs)
    n_val = max(1, len(songs) // 10) if len(songs) > 1 else 0
    val, train = songs[:n_val], songs[n_val:]

    mean, std = compute_norm(train)
    if name == "crnn":
        ds = ChunkDataset(train, mean, std, augment=augment, shifts=shifts)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
    else:
        ds = FrameDataset(train, mean, std, context=context, augment=augment, shifts=shifts)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = build_model(name, context, width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        val_acc = frame_accuracy(model, name, val or train, mean, std, context, device)
        history.append({"epoch": epoch, "train_loss": total / len(ds), "val_frame_acc": val_acc})
        if verbose:
            print(f"  epoch {epoch:2d}  loss {total / len(ds):.3f}  validation frame accuracy {val_acc:.3f}")
        if val_acc > best_acc:# keep the best epoch
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, mean, std, history
