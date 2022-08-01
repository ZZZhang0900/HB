from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import metrics
from statsmodels.graphics.tsaplots import plot_acf


######读取数据
#data = pd.read_csv(r"F:\pythonProject\data2\monthly_total_zhihou_time.csv.csv")      ##[1,2,3,4,5,6,8,9,10,11,12,13]'temp','preci','rh'滞后2个月的数据，'XJ','dailyNum'同期
x= pd.read_csv(r"F:\pythonProject\data2\monthly_total_zhihou.csv",usecols = [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20],nrows=60)
print(x)
y = pd.read_csv(r"F:\pythonProject\data2\monthly_total_zhihou.csv",usecols = [21],nrows=60)
print(y)
xy = pd.read_csv(r"F:\pythonProject\data2\monthly_total_zhihou.csv")



StandardScalerS = StandardScaler()      #标准化
x_transf2 = StandardScalerS.fit_transform(x)
y_transf2 = StandardScalerS.fit_transform(y)
#y = data.iloc[0:60, 37]
'''t = np.linspace(0,1,60)
print(x_transf2.shape)'''

x_train, x_test, y_train, y_test = train_test_split(x_transf2, y_transf2, test_size = 0.2, random_state = 0)


'''n_estimator_params = range(1, 1000,5)
confusion_matrixes = {}
t=0.2
for n_estimator in n_estimator_params:
    rf = RandomForestRegressor(n_estimators=n_estimator,n_jobs=-1, verbose=True)
    rf.fit(x_train, y_train)
    random_forest_predict = rf.predict(x_train)
    random_forest_RMSE = metrics.mean_squared_error(y_train, random_forest_predict) ** 0.5

    print ("Accuracy:\t", random_forest_RMSE,n_estimator)
    if random_forest_RMSE < t:
        t = random_forest_RMSE
        i=n_estimator
        print(t,i)
print(t,i)'''


feat_labels = x.columns
forest = RandomForestRegressor(n_estimators=641, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
random_forest_predict=forest.predict(x_test)
print(y_test.shape,random_forest_predict.shape)
random_forest_error=random_forest_predict - y_test


importances = forest.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


######################重要性排序图
'''a = pd.DataFrame({"feature": ['brucella', 'atmospheric pressure_0', 'relative humidity_2', 'precipitation_3',
                              'relative humidity_0', 'wind speed_1', 'atmospheric pressure_3', 'temperature_3', 'relative humidity_3']})
b = pd.DataFrame({"importance": [0.525600,  0.091605,  0.080211,  0.078621,  0.062494,  0.025102,  0.018400,  0.018379,  0.013365]})'''
a = pd.DataFrame({"feature": ['B', 'A_Lag0', 'P_Lag3', 'R_Lag2','R_Lag0','W_Lag1', 'A_Lag3', 'W_Lag0', 'R_Lag3']})
b = pd.DataFrame({"importance": [0.557911,  0.084800,  0.071919,  0.070662,  0.053096,  0.025950,  0.021823,  0.014051, 0.013686]})
df = pd.concat([a, b], axis=1)
# 对特征按照重要性程度的从大到小进行排序
df = df.sort_values(by="importance", ascending=False)
# 只输出前10个重要的特征
print(df)


f= plt.figure(figsize=(15,10))
ax = f.add_axes([0.2,0.15,0.7,0.7])
# 设置order参数：按重要程度（importance）从大到小输出的结果:
g = sns.barplot(x="importance", y="feature",
					data=df, order=df["feature"], orient="h")
##坐标轴刻度值属性设置：
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax.set_xlabel('Importance',fontsize=21, color='black',fontweight='bold',labelpad=4.5,fontname='Times New Roman')
ax.set_ylabel('Varibles',fontsize=21, color='black',fontweight='bold',labelpad=4.5,fontname='Times New Roman')

#sns.despine(bottom=False, left=False)  # 设置是否显示边界线

#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
plt.show()



# #######Draw test plot进行预测图像绘制，其中包括预测结果的拟合图与误差分布直方图。
plt.figure(1)
plt.clf()
ax=plt.axes(aspect='equal')
plt.scatter(y_test,random_forest_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')
Lims=[0,2000]
plt.xlim(Lims)
plt.ylim(Lims)
plt.plot(Lims,Lims)
plt.grid(False)
plt.show()

plt.figure(2)
plt.clf()
plt.hist(random_forest_error,bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.grid(False)
plt.show()


# Verify the accuracy用皮spearman、决定系数与RMSE作为精度的衡量指标
random_forest_pearson_r=stats.spearmanr(y_test,random_forest_predict)
random_forest_R2=metrics.r2_score(y_test,random_forest_predict)
random_forest_RMSE=metrics.mean_squared_error(y_test,random_forest_predict)**0.5
print('spearmanr  is {0}, R2 is{1}, and RMSE is {2}.'.format(random_forest_pearson_r[0],random_forest_R2,
                                                                        random_forest_RMSE))




