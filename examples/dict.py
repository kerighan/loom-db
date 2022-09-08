import loom
import numpy as np
from tqdm import tqdm

N = 100000
X = np.random.random(size=(N, 100))

db = loom.Dict("dict.loom", dtype="blob", flag="w", blob_protocol="pickle")
# for i in tqdm(range(N), desc="write"):
#     db[f"key_{i}"] = list(X[i])

for i in tqdm(range(N), desc="read"):
    db[f"key_{i}"]
