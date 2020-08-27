# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:18:36 2020

@author: Vimal PM
"""
#importing the libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

#invoking the dataset using pd.read_csv()

PlasticSales=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//Work done//Forecasting//PlasticSales.csv")
PlasticSales.columns
#preprocessing data's
PlasticSales['t']=0
for i in range(0,60):
    PlasticSales['t'][i]=i+1

PlasticSales["t_squared"]=PlasticSales["t"]*PlasticSales["t"]
PlasticSales["log_sales"]=np.log(PlasticSales["Sales"])
PlasticSales.Sales.plot()

#from the plot I can say that trend is linear and seasonality is multiplicative

p = PlasticSales["Month"][0]
p[0:3]
PlasticSales['months']= 0

for i in range(60):
    p = PlasticSales["Month"][i]
    PlasticSales['months'][i]= p[0:3]
    
    
#getting the dummies of the months variable
       
month_dummies = pd.DataFrame(pd.get_dummies(PlasticSales['months']))
PlasticSales = pd.concat([PlasticSales,month_dummies],axis = 1)

PlasticSales.columns
PlasticSales=PlasticSales.drop(["months"],axis=1)
#Index(['Month', 'Sales', 't', 't_squared', 'log_sales', 'Apr', 'Aug',
#'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep']


#splitting the train and test data's
Train=PlasticSales.head(48)
Test=PlasticSales.tail(12)

###linear model###
linear_model=smf.ols("Sales~t",data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test["t"])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


###Exponential model###
exp_model=smf.ols("log_sales~t",data=Train).fit()
pred_exp=pd.Series(exp_model.predict(pd.DataFrame(Test["t"])))
rmse_exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_exp)))**2))
rmse_exp


###Quadratic model###
Quad_model=smf.ols("Sales~t+t_squared",data=Train).fit()
pred_quad=pd.Series(Quad_model.predict(pd.DataFrame(Test[["t","t_squared"]])))
rmse_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_quad))**2))
rmse_quad

###Additive seasonality model###
Add_model=smf.ols("Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_add=pd.Series(Add_model.predict(pd.DataFrame(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]])))
rmse_add = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add))**2))
rmse_add


###Additive quadratic seasonal model###
AddQuad_model=smf.ols("Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_addquad=pd.Series(AddQuad_model.predict(pd.DataFrame(Test[["t","t_squared","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]])))
rmse_AddQuad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_addquad))**2))
rmse_AddQuad 

###Multiplicative seasonality model###
Mul_model=smf.ols("log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul=pd.Series(Mul_model.predict(pd.DataFrame(Test)))
rmse_mul = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mul)))**2))
rmse_mul

###Multiplicative Additive seasonal model###
MulAdd=smf.ols("log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_MulAdd=pd.Series(MulAdd.predict(Test))
rmse_MulAdd = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_MulAdd)))**2))
rmse_MulAdd 

#Creating a dictionary for rmse values
data={"Model":(["rmse_linear","rmse_exp","rmse_quad","rmse_add","rmse_AddQuad","rmse_mul","rmse_MulAdd"]),"RMSE_Values":([rmse_linear,rmse_exp,rmse_quad,rmse_add,rmse_AddQuad,rmse_mul,rmse_MulAdd])}
RMSE_Table=pd.DataFrame(data)
RMSE_Table
#          Model   RMSE_Values
#0   rmse_linear   260.937814
#1      rmse_exp   268.693839
#2     rmse_quad   297.406710
#3      rmse_add   235.602674
#4  rmse_AddQuad   218.193876
#5      rmse_mul   239.654321
#6   rmse_MulAdd   160.683329

#Above shows the RMSE values which I have got from each and every models..
#Based on these values I can say my Multiplicative additive seasonal  model is the best or significant model for further analysis which having less RMSE value(160).
