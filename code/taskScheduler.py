import pandas as pd
import numpy as np
import time
import datetime
import requests
import matplotlib.pyplot as plt
from client import engine
from greenLight import greenLight
from terminalLogic import Algo
import json
import schedule

def fetchData():
    print('..fetching BTCPERP OLHC data feed')
    data = engine.getData('BTCPERP')
    data = data[::-1]
    df = Algo.fullstate(data, data)
    df.dropna(inplace=True)
    algo = df
    algo['time'] = algo['time'].dt.date
    table = algo.tail(20)
    table.to_json('../web/templates/table.json', orient='records')
schedule.every(15).seconds.do(fetchData)
while True:
    schedule.run_pending()
    time.sleep(1)