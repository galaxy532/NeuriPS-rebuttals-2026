"""
extract_features.py -- ERM training and frozen-representation extraction.

Run this on a GPU machine. It trains an ERM classifier on Waterbirds (or a
CelebA-style dataset), freezes the feature extractor Phi, and caches the
penultimate activations together with the label y and the group index g into an
.npz bundle that every other script in this folder consumes.

Group definition and why Waterbirds maps onto the manuscript's setup
-------------------------------------------------------------------
The manuscript's two groups are distinguished by the SIGN of the alignment
between the spurious block and the causal signal:

    G_maj : s = A r + xi      G_min : s = B r + xi,      A != B.

In Waterbirds the label y is the bird type and the spurious attribute is the
background (`place`). The standard construction makes background agree with
bird type for ~95% of the training data and disagree for the rest. So

    g = 0 (majority) iff place agrees with y,
    g = 1 (minority) iff place disagrees with y,

which is exactly a sign reversal of the background-to-label alignment between
groups, i.e. the operational content of A != B. The resulting minority fraction
is eps ~ 0.05, and it is reported by this script rather than assumed.

A caution about CelebA. The usual CelebA setup (y = Blond_Hair, spurious =
Male) does NOT map onto the two-group structure as cleanly, because the
imbalance lives inside the blond class rather than in an overall alignment flip:
defining g by whether Male agrees with the majority blond-female pairing yields
eps ~ 0.45, which is not an imbalance at all. Waterbirds is therefore the
primary target; CelebA is supported but the group definition should be revisited
before its numbers are used in an argument.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def build_waterbirds(root: str, split: str):
    """Return (paths, y, g) for one Waterbirds split.

    Expects the standard release layout: `metadata.csv` at `root` with columns
    img_id, img_filename, y, split, place. split codes are 0=train, 1=val, 2=test.
    """
    import pandas as pd

    md = pd.read_csv(os.path.join(root, "metadata.csv"))
    code = {"train": 0, "val": 1, "test": 2}[split]
    md = md[md["split"] == code].reset_index(drop=True)
    paths = [os.path.join(root, f) for f in md["img_filename"].tolist()]
    y_raw = md["y"].to_numpy().astype(int)          # 1 = waterbird
    place = md["place"].to_numpy().astype(int)      # 1 = water background
    y = np.where(y_raw == 1, 1, -1)                 # to the +/-1 convention
    g = (y_raw != place).astype(int)                # 0 = agrees (maj), 1 = disagrees (min)
    return paths, y, g


def main() -> None:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing metadata.csv")
    ap.add_argument("--out", default="features_waterbirds.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class DS(Dataset):
        def __init__(self, paths, y, g, tf):
            self.paths, self.y, self.g, self.tf = paths, y, g, tf

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.tf(img), int(self.y[i] > 0), int(self.g[i])

    p_tr, y_tr, g_tr = build_waterbirds(args.root, "train")
    print(f"train n={len(p_tr)}  eps(minority)={g_tr.mean():.4f}")

    # ERM training of the full network.
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    d_feat = net.fc.in_features
    net.fc = nn.Linear(d_feat, 2)
    net = net.to(dev)

    dl = DataLoader(DS(p_tr, y_tr, g_tr, tf_train), batch_size=args.bs,
                    shuffle=True, num_workers=8, pin_memory=True)
    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd)
    lossf = nn.CrossEntropyLoss()
    net.train()
    for ep in range(args.epochs):
        tot = corr = 0
        run = 0.0
        for xb, yb, _ in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            out = net(xb)
            loss = lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * yb.size(0)
            corr += (out.argmax(1) == yb).sum().item()
            tot += yb.size(0)
        print(f"  epoch {ep + 1}/{args.epochs}  loss={run / tot:.4f}  acc={corr / tot:.4f}")

    # Freeze Phi and cache penultimate activations.
    net.fc = nn.Identity()
    net.eval()
    paths, y, g = build_waterbirds(args.root, args.split)
    dl_e = DataLoader(DS(paths, y, g, tf_eval), batch_size=args.bs,
                      shuffle=False, num_workers=8, pin_memory=True)
    feats = []
    with torch.no_grad():
        for xb, _, _ in dl_e:
            feats.append(net(xb.to(dev, non_blocking=True)).cpu().numpy())
    phi = np.concatenate(feats).astype(np.float64)
    print(f"extracted Phi: {phi.shape}   split={args.split}   eps={g.mean():.4f}")

    np.savez_compressed(
        args.out, phi=phi, y=y.astype(int), g=g.astype(int),
        idx_r=np.arange(0), idx_s=np.arange(0),
        meta=np.array(repr({"source": "waterbirds", "split": args.split,
                            "eps": float(g.mean()), "d": int(phi.shape[1])}),
                      dtype=object),
    )
    print(f"wrote {args.out}  (idx_r/idx_s are filled in by analyze.py)")


if __name__ == "__main__":
    main()
