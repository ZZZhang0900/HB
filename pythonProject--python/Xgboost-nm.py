from numpy import loadtxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy import sort
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
import matplotlib.pyplot as plt
import shap
######读取数据
x= pd.read_csv(r"F:\pythonProject\data2\monthly_total_zhihou.csv",usecols = [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20],nrows=60)
print(x)
y = pd.read_csv(r"F:\pythonProject\data2\monthly_total_zhihou.csv",usecols = [21],nrows=60)
print(y)
xy = pd.read_csv(r"F:\pythonProject\data2\monthly_total_zhihou.csv")


StandardScalerS = StandardScaler()      #标准化
x_transf2 = StandardScalerS.fit_transform(x)
y = xy.iloc[0:60, 21]
print(y)

X_train, X_test, y_train, y_test = train_test_split(x_transf2, y, test_size=0.2, random_state=8)
feat_labels = x.columns
# fit model on all training data
model = XGBRegressor()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
'''predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBRegressor()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


plot_importance(model)
plt.show()
'''
importances = model.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


explainer = shap.Explainer(model)
shap_values = explainer(x)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])