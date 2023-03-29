import loom
import numpy as np
from tqdm import tqdm
from loom.datastructure.array import List

db = loom.DB("test.loom", flag="n")
data = db.create_dataset("data", value="uint64")
array = db.create_datastructure("array", List(data))
db.compile()

n = 50000
for i in tqdm(range(n), desc="insert"):
    array.append({"value": i})

for i in tqdm(range(n), desc="read"):
    assert array[i]["value"] == i
