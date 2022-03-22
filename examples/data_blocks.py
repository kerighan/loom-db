import loom
import numpy as np
from tqdm import tqdm

db = loom.DB("test.loom")
vectors = db.create_dataset("vectors", data="(200,)float32")
db.compile()

N = 1000
X = np.random.random(size=(N, 100))
block_id = vectors.new_block(N)
for i in tqdm(range(N)):
    vectors[block_id, i] = X[i]
