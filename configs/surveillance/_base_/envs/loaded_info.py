import pickle
from typing import List

with open("output/surv_room/SurveillanceRoom.pkl", "rb") as f:
    df = pickle.load(f)
SHAPE: List[int] = df["shape"]
