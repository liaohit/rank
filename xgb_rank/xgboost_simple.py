from sklearn import datasets
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

data_X, data_Y = datasets.load_svmlight_file("data/train/model_in.txt")
data_groups = []
with open("data/model_in.group") as fin:
    for line in fin:
        data_groups.append(int(line.rstrip()))

model = xgb.XGBRanker(
    booster='gbtree',
    objective='rank:pairwise',
    random_state=42,
    learning_rate=0.1,
    colsample_bytree=0.9,
    eta=0.05,
    max_depth=6,
    n_estimators=110,
    subsample=0.75
    )


model.fit(data_X, data_Y, group=data_groups)

model_file = "model.dat"
model.save_model(model_file)
