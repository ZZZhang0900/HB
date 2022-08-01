# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:02:34 2021

@author: lenovo
"""

#%%导入模块

#from pandas import Series

from matplotlib.pyplot import MultipleLocator
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import os
import math
import seaborn as sns
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

####################
temp = pd.read_csv(r'F:\pythonProject\data2\monthly_temp.csv')      ##'temp','preci','rh'滞后2个月的数据，'XJ','dailyNum'同期
preci = pd.read_csv(r"F:\pythonProject\data2\monthly_TotalPrecipitation.csv")['tp']
RH = pd.read_excel(r'F:\pythonProject\data2\RH.xlsx', sheet_name='Sheet1')
WS = pd.read_csv(r'F:\pythonProject\data2\monthly_WindSpeed.csv')['si10']
#UV = pd.read_csv(r"F:\pythonProject\data2\monthly_UV_mean.csv")["uvb"]
SP = pd.read_csv(r"F:\pythonProject\data2\monthly_SurfacePressure.csv")["sp"]
XiJun = pd.read_csv(r"F:\pythonProject\data2\XiJun.csv")
dailyNum = pd.read_excel(r'F:\pythonProject\data2\neimeng.xlsx', sheet_name='Sheet2')['incidence']   #发病率
print(dailyNum)
frames = [temp['time'],temp['temp'],preci, RH['RH'],WS,SP,XiJun,dailyNum]
frame1 = pd.concat(frames, axis=1)  # axis = 0 就是按行合并，你也可以去掉原来的列名，就变成
frame1 = pd.concat(frames, ignore_index=True, axis=1)

frame1.to_csv("F:\pythonProject\data2\monthly_total.csv",header = ['time','temp','preci','rh','ws','sp','XJ','dailyNum'])
data = pd.read_csv(r'F:\pythonProject\data2\monthly_total.csv')      ##'temp','preci','rh'滞后2个月的数据，'XJ','dailyNum'同期


####################全部滞后(无时间)####################
x1 = pd.read_csv(r'F:\pythonProject\data2\monthly_total.csv',usecols = [1,2,3,4,5,6]).shift(-3)   #'temp','preci','rh' 'XJ','dailyNum'同期
x2 = pd.read_csv(r'F:\pythonProject\data2\monthly_total.csv',usecols = [2,3,4,5,6]).shift(-2)     #滞后一个月
x3 = pd.read_csv(r'F:\pythonProject\data2\monthly_total.csv',usecols = [2,3,4,5,6]).shift(-1)     #滞后2个月
x4 = pd.read_csv(r'F:\pythonProject\data2\monthly_total.csv',usecols = [2,3,4,5,6])     #滞后3个月
x10= pd.read_csv(r'F:\pythonProject\data2\monthly_total.csv',usecols = [7,8])

frames = [x1['temp'], x1['preci'], x1['rh'], x1['ws'], x1['sp'],  x2, x3, x4, x10]
frame1 = pd.concat(frames, axis=1)  # axis = 0 就是按行合并，你也可以去掉原来的列名，就变成
frame1 = pd.concat(frames, ignore_index=True, axis=1)
#print(frame1)
'''frame1.to_csv("F:\pythonProject\data2\monthly_total_zhihou.csv",header =['temperature_0','precipitation_0','relative humidity_0','wind speed_0','atmospheric pressure_0',
                                                                        'temperature_1','precipitation_1','relative humidity_1','wind speed_1','atmospheric pressure_1',
                                                                        'temperature_2','precipitation_2','relative humidity_2','wind speed_2','atmospheric pressure_2',
                                                                        'temperature_3','precipitation_3','relative humidity_3','wind speed_3','atmospheric pressure_3',
                                                                        'brucella','dailyNum'],index=None)
'''
frame1.to_csv("F:\pythonProject\data2\monthly_total_zhihou.csv",header =['T_Lag0','P_Lag0','R_Lag0','W_Lag0','A_Lag0',
                                                                        'T_Lag1','P_Lag1','R_Lag1','W_Lag1','A_Lag1',
                                                                        'T_Lag2','P_Lag2','R_Lag2','W_Lag2','A_Lag2',
                                                                        'T_Lag3','P_Lag3','R_Lag3','W_Lag3','A_Lag3',
                                                                        'B','dailyNum'],index=None)

