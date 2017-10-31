# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 08:39:51 2017

@author: Aishu
"""

import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For time stamps
from datetime import datetime

from __future__ import division

# Apple Stocks Historical data 
AAPL = pd.read_csv("C:\\Users\\Aishu\\Desktop\\My_Projects\\Stock Analysis\\AAPL.csv")
AAPL.dtypes
AAPL.describe()
AAPL.head()

dates = AAPL["Date"]
date_objects = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

# plotting stocks data with time 
plt.plot(date_objects,AAPL["Adj Close"])
plt.plot(date_objects,AAPL["Volume"])

# Moving averages 
moving_10 = pd.rolling_mean(AAPL["Adj Close"],10)
plt.plot(date_objects, moving_10)


AAPL["Daily Return"] = AAPL["Adj Close"].pct_change()
plt.plot(date_objects, AAPL["Daily Return"],marker = 'o',linestyle='--')
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')

#DMart Stocks 
DMART = pd.read_csv("C:\\Users\\Aishu\\Desktop\\My_Projects\\Stock Analysis\\DMART.NS.csv")

date1 = DMART["Date"]
date_object = [datetime.strptime(date, '%Y-%m-%d').date() for date in date1]

#plotting stocks with time 
plt.plot(date_object,DMART["Adj Close"])
plt.plot(date_object,DMART["Volume"])
DMART["Daily Return"] = DMART["Adj Close"].pct_change()
plt.plot(date_object, DMART["Daily Return"],marker = 'o',linestyle='--')
sns.distplot(DMART['Daily Return'].dropna(),bins=100,color='purple')

#Risk Analysis 
rets = DMART["Daily Return"]

plt.scatter(rets.mean(),rets.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
#value at risk 
rets.quantile(0.05)


# Monte Carlo Simulations 

days = 92
dt = 1/days 
mu = rets.mean()
sigma = rets.std()

def Stock_monte_carlo(start_price,days,mu,sigma):
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in xrange(1,days):
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price

start_price = 891.3

for run in xrange(100):
    plt.plot(Stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Dmart')

for run in xrange(10000):
    simulations = Stock_monte_carlo(start_price,days,mu,sigma)
plt.hist(simulations,bins =50)

print simulations.mean()
print simulations.std()

Series(simulations).quantile(1)

