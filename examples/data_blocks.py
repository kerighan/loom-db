import loom
import numpy as np
from tqdm import tqdm

N = 1000


def create_database():
    # create database
    db = loom.DB("test.loom", flag="n")
    vectors = db.create_dataset("vectors", data="(100,)float32")
    db.compile()

    # generate data
    X = np.random.random(size=(N, 100))

    block_id = vectors.new_block(N)
    print(block_id)
    for i in tqdm(range(N)):
        vectors[block_id, i] = {"data": X[0]}


def read_database():
    db = loom.DB("test.loom", flag="r")
    vectors = db["vectors"]

    block_id = 541
    for i in tqdm(range(N)):
        print(vectors[block_id, i, "data"])


create_database()
read_database()
