import sys
import matplotlib.pyplot as plt
from skimage.io import imread

name = sys.argv[1]
dirs = ["data/SEGTHOR", "data/SEGTHOR_MED", "data/SEGTHOR_WIDE"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, d in zip(axes, dirs):
    ax.imshow(imread(f"{d}/val/img/{name}"), cmap="gray", vmin=0, vmax=255)
    ax.set_title(d)
    ax.axis("off")
plt.show()