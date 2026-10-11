import sys
import numpy as np

for path in sys.argv[1:]:
    d = np.load(f"{path}/dice_val.npy")       # shape: (epochs, slices, classes)
    per_class = d[:, :, 1:].mean(axis=1)       # mean over slices -> (epochs, 4)
    best = per_class.mean(axis=1).argmax()
    last = per_class[-1]
    print(f"{path:20s} best ep {best}: {per_class[best].round(3)} mean {per_class[best].mean():.3f}"
          f" | last: {last.round(3)} mean {last.mean():.3f}")
