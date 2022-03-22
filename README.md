loom.DB
=======

How to install:

```
pip install loom-db
```

How to use:

```python
import loom
import numpy as np
from tqdm import tqdm

N = 100000
X = np.random.random(size=(N, 100))

db = loom.Dict("dict.loom", dtype="(100,)float32")

# insert data
for i in tqdm(range(N), desc="write"):
    db[f"key_{i}"] = X[i]

# get data
for i in tqdm(range(N), desc="read"):
    db[f"key_{i}"]
```
