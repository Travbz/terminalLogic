import requests
import matplotlib.pyplot as plt
# import terminalLogic
# from terminalLogic import Algo
# import client
# from client import engine
# from typing import Optional, Dict, Any, List
# from ciso8601 import parse_datetime
from requests import Request, Session, Response
import pandas as pd
import numpy as np
import datetime
import time
from time import sleep 
import sched
btc1m =  f'https://ftx.com/api/markets/BTC/USD/candles?resolution=60'
dupe = pd.DataFrame(columns=['startTime', 'time', 'open', 'high', 'low', 'close', 'volume'])
s = sched.scheduler(time.time, time.sleep)
seconds = time.time()
local_time = time.ctime(seconds)


def pullData():
    print('-------------------------------------------------------------------_-')
    print('...initiating REST API call to FTX Exchange historical data endpoints')

    try:
        r = requests.get(btc1m,timeout=3)
        r.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)    
    minute = pd.json_normalize(r.json()['result'])
    minute = minute.iloc[-1:, :]
    dupe = pd.DataFrame(columns=['startTime', 'time', 'open', 'high', 'low', 'close', 'volume'])
    dupe = dupe.append(minute)
    dupe.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    return dupe
def writeDB():
    print('---------------------------------------------------------------------')
    print("...writing a row to the .csv")
    return dupe.to_csv("minute.csv")

def normalizeJSON():
    print('---------------------------------------------------------------------')
    print('...harvesting and normalizing JSON Object')
def runTest():
    print('---------------------------------------------------------------------')
    print('...running all backtests and ML sandbox')
    
def state():
    print('---------------------------------------------------------------------')
    print('..desyphering market state')
    
def algg():
    print('---------------------------------------------------------------------')
    print('..market state selected, algo implimented for next period')
    
def positions():
    print('---------------------------------------------------------------------')
    print('..generating signals for positions')

def order():
    print('---------------------------------------------------------------------')
    print('...placing all orders')
    
def toSQL():
    print('---------------------------------------------------------------------')
    print('...writing new rows to postgreSQL DB')

def logs():
    print('---------------------------------------------------------------------')
    print('..generating portfolio logs and updating PnL')
    
def wait():
    print('---------------------------------------------------------------------')
    print('...waiting for next period to make decisions on new data')

def restart():
    print('---------------------------------------------------------------------')
    print('repeating data retrieval steps')
    print('---------------------------------------------------------------------')
    print(local_time)
    print('---------------------------------------------------------------------')
    print('---------------------------------------------------------------------')
    
def runScript():
    print(local_time)
    print('---------------------------------------------------------------------')
    print("Spinning up terminalLogic's data retrieval protocol now.. ")
    while True:
        s.enter(1, 1, pullData)
        s.enter(2,2,normalizeJSON)
        s.enter(3, 1, writeDB)
        s.enter(4,3, toSQL)
        s.enter(10,2,restart)
        s.enter(5,1,runTest)
        s.enter(6,2, positions)
        s.enter(7,2,order)
        s.enter(8,2,state)
        s.enter(9,2,algg)



        s.run()
        time.sleep(1)
a = runScript()
a