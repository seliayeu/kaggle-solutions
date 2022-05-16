import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

test_data = pd.read_csv("./data/test.csv")
train_data = pd.read_csv("./data/train.csv")

test_X = test_data.select_dtypes(include=["number"])
test_X = test_X.fillna(test_X.mean())

train_X = train_data.select_dtypes(include=["number"])
train_X = train_X.fillna(train_X.mean())
train_X = train_X.drop(["SalePrice"], axis="columns")
train_y = train_data.SalePrice

model = RandomForestRegressor()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(train_X, train_y)

features = []

for i in range(len(fit.support_)):
    if fit.support_[i]: features.append(list(train_X)[i])

train_X = train_data[features]
train_X = train_X.fillna(train_X.mean())

model = RandomForestRegressor(random_state=37)
model.fit(train_X, train_y)

predictions = pd.DataFrame({ "SalePrice": model.predict(test_X[features])})
predictions["Id"] = test_X["Id"]
predictions.to_csv("./submission.csv", index=False)