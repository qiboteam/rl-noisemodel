from pathlib import Path
import numpy as np
sets_name = Path(__file__).parent.glob("*.npz")
for path in sets_name:
    data = np.load(path, allow_pickle=True)
    print(data["clean_rep"].shape)

