#!/usr/bin/env python3
"""Sanity check: can the model memorise a handful of samples?

Not a measure of quality — only proof that gradients reach every parameter
and that the model has enough capacity to fit at all.
"""

import torch
import torch.nn.functional as F

from ViT import ViT
from losses import CrossEntropy

torch.manual_seed(0)

B, K = 4, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = ViT(1, K, dim=96, depth=2).to(device)
net.init_weights()

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
loss_fn = CrossEntropy(idk=list(range(K)))

# Fixed random batch: image and a one-hot target that the network must memorise.
img = torch.rand(B, 1, 256, 256, device=device)
labels = torch.randint(0, K, (B, 256, 256), device=device)
gt = F.one_hot(labels, num_classes=K).permute(0, 3, 1, 2).float()

for i in range(200):
    optimizer.zero_grad()
    pred_probs = F.softmax(net(img), dim=1)
    loss = loss_fn(pred_probs, gt)
    loss.backward()
    optimizer.step()

    if i % 20 == 0 or i == 199:
        print(f"{i:4d}  loss {loss.item():.4f}")