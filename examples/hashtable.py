import loom
import numpy as np
from loom.datastructure import Hashmap

db = loom.DB("test.loom", flag="n")
data = db.create_dataset("data", key="U15", value="(100,)float32")
table = db.create_datastructure("table", Hashmap(data, "key"))
db.compile()

table["test"] = {"value": np.random.random(size=100)}
print(table["test"])
