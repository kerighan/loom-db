import loom
import numpy as np
from loom.datastructure import Hashmap
from loom.datastructure.hashtable import CompactHashmap

db = loom.DB("test.loom", flag="n", blob_compression="zlib")
data = db.create_dataset("data", value="uint64")
table = db.create_datastructure("table", CompactHashmap(data))
print("here")
db.compile()

table["test"] = {"value": 15}
table["test"] = {"value": 9}
# for i in range(500):
#     table[f"test_{i}"] = {"value": i}
# table["test"] = {"value": ["blob"]}
# print(table["test"])
# table["test2"] = {"value": ["blob2"]}
# print(table["test2"])
