# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:56:50 2020

@author: Vimal PM

"""

#importing the libraries 
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#loading the dataaset
cocacola=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//Work done//Forecasting//CocaCola_Sales_Rawdata.csv")
cocacola.columns
#Index(['Quarter', 'Sales'], dtype='object')
#adding the time period
cocacola["t"]=0
for i in range(42):
    cocacola["t"][i]=i+1

#addding the logrothmic sales and tsquared values.
cocacola["t_squared"]=cocacola["t"]*cocacola["t"]    
cocacola["log_sales"]=np.log(cocacola["Sales"])

cocacola.Sales.plot()
#from the plot I can say my trend is exponential and seasonality is multiplicative.

#splitting the Quarter varaiabe.
p=cocacola["Quarter"][0]
p[0:3]
cocacola["quarters"]=0
for i in range(42):
    p=cocacola["Quarter"][i]
    cocacola["quarters"][i]=p[0:3]

#getting the dummies of Quarter variables    
Quarter_dummies=pd.DataFrame(pd.get_dummies(cocacola["quarters"]))
cocacola=pd.concat([cocacola,Quarter_dummies],axis=1)
cocacola=cocacola.drop(["quarters"],axis=1)
cocacola.columns
#Index(['Quarter', 'Sales', 't', 't_squared', 'log_sales', 'Q1_', 'Q2_', 'Q3_','Q4_']
#splitting the train and test data's
Train=cocacola.head(38)
Test=cocacola.tail(4)
    ####Model building####

###Building the linear model###
linear_model=smf.ols("Sales~t",data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test["t"])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

###Building the exponential model###
exp_model=smf.ols("log_sales~t",data=Train).fit()
pred_exp=pd.Series(exp_model.predict(pd.DataFrame(Test["t"])))
rmse_exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_exp)))**2))
rmse_exp

###Building the Quadratic model###
Quad_model=smf.ols("Sales~t+t_squared",data=Train).fit()
pred_quad=pd.Series(Quad_model.predict(pd.DataFrame(Test[["t","t_squared"]])))
rmse_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_quad))**2))
rmse_quad


###Building the Additive seasonality model###
Add_model=smf.ols("Sales~Q1_+Q2_+Q3_+Q4_",data=Train).fit()
pred_add=pd.Series(Add_model.predict(pd.DataFrame(Test[["Q1_","Q2_","Q3_","Q4_"]])))
rmse_add = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add))**2))
rmse_add

###Building the additive seasonal quadratic model###
AddQuad_model=smf.ols("Sales~t+t_squared+Q1_+Q2_+Q3_+Q4_",data=Train).fit()
pred_AddQuad=pd.Series(AddQuad_model.predict(pd.DataFrame(Test[["t","t_squared","Q1_","Q2_","Q3_","Q4_"]])))
rmse_AddQuad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_AddQuad))**2))
rmse_AddQuad 


###Building the Multiplicative seasonality###
Mul_model=smf.ols("log_sales~Q1_+Q2_+Q3_+Q4_",data=Train).fit()
pred_mul=pd.Series(Mul_model.predict(pd.DataFrame(Test)))
rmse_mul = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul)))**2))
rmse_mul

###Building the Multiplicative additive seasonality model###
MulAdd_model=smf.ols("log_sales~t+Q1_+Q2_+Q3_+Q4_",data=Train).fit()
pred_Muladd=pd.Series(MulAdd_model.predict(pd.DataFrame(Test)))
rmse_MulAdd = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Muladd)))**2))
rmse_MulAdd 
#Creating a dictionary for rmse values 
data={"Model":pd.Series(["rmse_linear","rmse_exp", "rmse_quad","rmse_add","rmse_AddQuad","rmse_mul","rmse_MulAdd"]),"RMSE_Values":pd.Series([rmse_linear,rmse_exp, rmse_quad,rmse_add,rmse_AddQuad,rmse_mul,rmse_MulAdd])}
RMSE_Table=pd.DataFrame(data)
RMSE_Table
#          Model  RMSE_Values
#0   rmse_linear   591.553296
#1      rmse_exp   466.247973
#2     rmse_quad   475.561835
#3      rmse_add  1860.023815
#4  rmse_AddQuad   301.738007
#5      rmse_mul  1963.389640
#6   rmse_MulAdd   225.524391

#From this analysis I found Multiplicative Additive  model is the best or significant model for the further analysis which having less RMSE value(225)
