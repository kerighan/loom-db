import loom
import numpy as np
from tqdm import tqdm

N = 100000
X = np.random.random(size=(N, 100))

db = loom.Dict("dict.loom", dtype="(100,)float32", flag="w")
for i in tqdm(range(N), desc="write"):
    db[f"test_{i}"] = X[i]

for i in tqdm(range(N), desc="read"):
    db[f"test_{i}"]
