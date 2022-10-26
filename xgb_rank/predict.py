import xgboost as xgb
import sys
import numpy as np
from xgboost import DMatrix
import pandas as pd

bst = xgb.Booster()
bst.load_model("model.dat")
datas = []
feature_list = []

for line in open("data/predict/model_in.txt"):
    elems = line.rstrip().split('\t')
    feats = np.array([float(x.split(':')[1]) for x in elems[1:]], dtype=float)
    datas.append(feats)
    feature_list = ["F_%d" % i for i in range(len(datas[0]))]

data_dict = {}
for i in range(len(datas)):
    for j in range(len(datas[0])):
        if j == 0:
            if "qid" not in data_dict:
                data_dict["qid"] = [datas[i][j]]
            else:
                data_dict["qid"].append(datas[i][j])
        if j != 1:
            if feature_list[j - 1] not in data_dict:
                data_dict[feature_list[j - 1]] = [datas[i][j]]
            else:
                data_dict[feature_list[j - 1]].append(datas[i][j])

# print(data_dict)

df = pd.DataFrame(data_dict)

def predict(model, df):
    data = df.loc[:, ~df.columns.isin(['qid'])]
    return model.predict(DMatrix(np.array(data), missing=np.NaN))

# predict by group
predictions = (df.groupby('qid')
               .apply(lambda x: predict(bst, x)))

print(predictions)

