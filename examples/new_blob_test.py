import loom
from tqdm import tqdm
from loom.datastructure.array import List

db = loom.DB("test.loom", flag="n")
dataset = db.create_dataset("test", value="uint64", blob="blob")
array = db.create_datastructure("test", List(dataset))
db.compile()

for i in tqdm(range(50000)):
    array.append({"value": i, "blob": bytes(f"test_{i}", "utf8")})

for i in tqdm(range(50000)):
    array[i]
