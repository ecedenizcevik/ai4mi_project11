import numpy as np

for name in ["SEGTHOR", "SEGTHOR_MED", "SEGTHOR_WIDE"]:
    d = np.load(f"results/hu/{name}/dice_val.npy")   # shape: (epochs, slices, classes)
    per_class = d[:, :, 1:].mean(axis=1)              # mean over slices -> (epochs, 4)
    best = per_class.mean(axis=1).argmax()
    print(f"{name:13s} epoch {best}  per class {per_class[best].round(3)}  mean {per_class[best].mean():.3f}")