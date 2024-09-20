import pickle

with open("output/surv_room/SurveillanceRoom.pkl", "rb") as f:
    df = pickle.load(f)
SHAPE = df["shape"]
