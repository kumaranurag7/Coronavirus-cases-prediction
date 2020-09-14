import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv('confirmed cases.csv')

df.fillna(0, inplace = True)

dates = df.columns.values[1:]
days_since = np.array([i for i in range(len(dates))]).reshape(-1,1)

daysinfuture = 49
future_forecast = np.array([i for i in range(len(dates)+daysinfuture)]).reshape(-1,1)
adjusted_dates = future_forecast[:-49]

start = '1/30/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days = i)).strftime('%m/%d/%Y'))


cases = df.iloc[0,:].values[1:]
total_cases = np.array(cases).reshape(-1,1)
    
xtrain,xtest,ytrain,ytest = train_test_split(days_since,total_cases, test_size = 0.3, shuffle = False)

poly = PolynomialFeatures(degree = 3)
polyx = poly.fit_transform(days_since)
polyforecast = poly.transform(future_forecast)
polyxtest= poly.transform(xtest)
        

lr = LinearRegression(fit_intercept = False)#
lr.fit(polyx, total_cases)
ypred = lr.predict(polyxtest)
ypredforecast = lr.predict(polyforecast)
r2 = r2_score(ytest,ypred)
print(r2, 'jj')
#print(future_forecast)  
'''
test1 = [[244]] 
test2 = poly.transform(test1)
print(test2)
pred11 = lr.predict(test2)
print(pred11)
'''    

pickle.dump(lr, open('pred.pkl', 'wb'))
pickle.dump(poly, open('transform.pkl', 'wb'))


for i in range(1,36):
    cases = df.iloc[i,:].values[1:]
    total_cases = np.array(cases).reshape(-1,1)
    
    xtrain,xtest,ytrain,ytest = train_test_split(days_since,total_cases, test_size = 0.3, shuffle = False)
    finalforecast = []
    tempr2 = 0
    temp = 0
    for j in range(2,4):
        poly = PolynomialFeatures(degree = j)
        polyx = poly.fit_transform(days_since)
        polyforecast = poly.transform(future_forecast)
        polyxtest= poly.transform(xtest)
        

        lr = LinearRegression(fit_intercept = False)#
        lr.fit(polyx, total_cases)
        ypred = lr.predict(polyxtest)
        ypredforecast = lr.predict(polyforecast)
        r2 = r2_score(ytest,ypred)
        
        if r2 > tempr2:
            tempr2 = r2
            finalforecast = ypredforecast
            temp = j
            
        if tempr2 ==0:
            lr2 = LinearRegression(fit_intercept = False)
            lr2.fit(days_since,total_cases)
            ypred = lr2.predict(xtest)
            finalforecast = lr2.predict(future_forecast)
            r2 = r2_score(ytest,ypred)
            
    if i == 1:
        pickle.dump(lr, open('an.pkl', 'wb'))
    elif i ==2:
        pickle.dump(lr, open('ap.pkl', 'wb'))
    elif i ==3:
        pickle.dump(lr, open('ar.pkl', 'wb'))
    elif i ==4:
        pickle.dump(lr, open('as.pkl', 'wb'))
    elif i ==5:
        pickle.dump(lr, open('br.pkl', 'wb'))
    elif i ==6:
        pickle.dump(lr, open('ch.pkl', 'wb'))
    elif i ==7:
        pickle.dump(lr, open('ct.pkl', 'wb'))
    elif i ==8:
        pickle.dump(lr, open('dndd.pkl', 'wb'))
    elif i ==9:
        pickle.dump(lr, open('dl.pkl', 'wb'))
    elif i ==10:
        pickle.dump(lr, open('ga.pkl', 'wb'))
    elif i ==11:
        pickle.dump(lr, open('gj.pkl', 'wb'))
    elif i ==12:
        pickle.dump(lr, open('hr.pkl', 'wb'))
    elif i ==13:
        pickle.dump(lr, open('hp.pkl', 'wb'))
    elif i ==14:
        pickle.dump(lr, open('jk.pkl', 'wb'))
    elif i ==15:
        pickle.dump(lr, open('jh.pkl', 'wb'))
    elif i ==16:
        pickle.dump(lr, open('ka.pkl', 'wb'))
    elif i ==17:
        pickle.dump(lr, open('kl.pkl', 'wb'))
    elif i ==18:
        pickle.dump(lr, open('la.pkl', 'wb'))
    elif i ==19:
        pickle.dump(lr, open('mp.pkl', 'wb'))
    elif i ==20:
        pickle.dump(lr, open('mh.pkl', 'wb'))
    elif i ==21:
        pickle.dump(lr, open('mn.pkl', 'wb'))
    elif i ==22:
        pickle.dump(lr, open('ml.pkl', 'wb'))
    elif i ==23:
        pickle.dump(lr, open('mz.pkl', 'wb'))
    elif i ==24:
        pickle.dump(lr, open('nl.pkl', 'wb'))
    elif i ==25:
        pickle.dump(lr, open('or.pkl', 'wb'))
    elif i ==26:
        pickle.dump(lr, open('py.pkl', 'wb'))
    elif i ==27:
        pickle.dump(lr, open('pb.pkl', 'wb'))
    elif i ==28:
        pickle.dump(lr, open('rj.pkl', 'wb'))
    elif i ==29:
        pickle.dump(lr, open('sk.pkl', 'wb'))
    elif i ==30:
        pickle.dump(lr, open('tn.pkl', 'wb'))
    elif i ==31:
        pickle.dump(lr, open('tg.pkl', 'wb'))
    elif i ==32:
        pickle.dump(lr, open('tr.pkl', 'wb'))
    elif i ==33:
        pickle.dump(lr, open('up.pkl', 'wb'))
    elif i ==34:
        pickle.dump(lr, open('ut.pkl', 'wb'))
    else:
        pickle.dump(lr, open('wb.pkl', 'wb'))

        
        
        
        
        
        
        
        
        
        
        
        
        


