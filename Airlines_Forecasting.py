# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:59:32 2020

@author: Vimal PM
"""
#importing necessary libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#Invoking the dataset using pd.read_csv()

Airlines=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//Work done//Forecasting//Airlines+Data.csv")

Airlines["t"]=0
for i in range(96):
    Airlines["t"][i]=i+1

Airlines["t_squared"]=Airlines["t"]*Airlines["t"]    

Airlines["log_passengers"]=np.log(Airlines["Passengers"])

Airlines.Passengers.plot()
#from the plot I can say that trend of passengers is linear and seasonality is multiplicative
p=Airlines["Month"][0]
p[0:3]
Airlines["months"]=0
for i in range(96):
    p=Airlines["Month"][i]
    Airlines["months"][i]=p[0:3]
#getting the dummies of month variable
month_dummies=pd.DataFrame(pd.get_dummies(Airlines["months"])) 
Airlines=pd.concat([Airlines,month_dummies],axis=1)   
Airlines.columns
#Index(['Month', 'Passengers', 't', 't_squared', 'log_passengers', 'months',   'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep']

#splitting the train test data's
Train=Airlines.head(84)
Test=Airlines.tail(12)
###Building the models###

###linear model###
linear_model=smf.ols("Passengers~t",data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test["t"])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


###Exponential model###
exp_model=smf.ols("log_passengers~t",data=Train).fit()
pred_exp=pd.Series(exp_model.predict(pd.DataFrame(Test["t"])))
rmse_exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_exp)))**2))
rmse_exp


###Quadratic model###
Quad_model=smf.ols("Passengers~t+t_squared",data=Train).fit()
pred_quad=pd.Series(Quad_model.predict(pd.DataFrame(Test[["t","t_squared"]])))
rmse_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_quad))**2))
rmse_quad

###Additive seasonality model###
Add_model=smf.ols("Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_add=pd.Series(Add_model.predict(pd.DataFrame(Test[["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]])))
rmse_add = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add))**2))
rmse_add


###Additive quadratic seasonal model###
AddQuad_model=smf.ols("Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_addquad=pd.Series(AddQuad_model.predict(pd.DataFrame(Test[["t","t_squared","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]])))
rmse_AddQuad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_addquad))**2))
rmse_AddQuad 

###Multiplicative seasonality model###
Mul_model=smf.ols("log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul=pd.Series(Mul_model.predict(pd.DataFrame(Test)))
rmse_mul = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_mul)))**2))
rmse_mul

###Multiplicative Additive seasonal model###
MulAdd=smf.ols("log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_MulAdd=pd.Series(MulAdd.predict(Test))
rmse_MulAdd = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_MulAdd)))**2))
rmse_MulAdd 

#Creating a dictionary for rmse values
data={"Model":(["rmse_linear","rmse_exp","rmse_quad","rmse_add","rmse_AddQuad","rmse_mul","rmse_MulAdd"]),"RMSE_Values":([rmse_linear,rmse_exp,rmse_quad,rmse_add,rmse_AddQuad,rmse_mul,rmse_MulAdd])}
RMSE_Table=pd.DataFrame(data)
RMSE_Table
#          Model   RMSE_Values
#0   rmse_linear    53.199237
#1      rmse_exp    46.057361
#2     rmse_quad    48.051889
#3      rmse_add   132.819785
#4  rmse_AddQuad    26.360818
#5      rmse_mul   140.063202
#6   rmse_MulAdd    10.519173

#From this analysis I can say that My Multiplicative Additive model is the best or significant model for further process which having less RMSE values(10).
