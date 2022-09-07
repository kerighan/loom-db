import loom
import numpy as np
from loom.datastructure import Hashmap

db = loom.DB("test.loom", flag="n", blob_compression="zlib")
data = db.create_dataset("data", key="U15", value="blob")
table = db.create_datastructure("table", Hashmap(data, "key"))
db.compile()

blob = {f"key_{i}": i for i in range(1000)}
table["test"] = {"value": blob}
# get data
table["test"]
