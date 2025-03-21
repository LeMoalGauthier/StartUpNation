import pandas as pd

df = pd.read_csv("evaluation.csv", sep=",")

scores = df[["FID", "CLIP"]]
print(scores)