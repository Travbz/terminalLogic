import pandas as pd
import numpy as np
import time
import datetime
import requests
import matplotlib.pyplot as plt
from client import engine
from terminalLogic import Algo
import schedule
plt.rcParams['figure.figsize'] = [16.0, 6.0]

while True:
    print('===running all protocols===')
    data = engine.getData('BTCPERP')
    df = Algo.fullstate(data, data)
    df.dropna(inplace=True)
    algo = df
    algo['just_time'] = algo['time'].dt.date
    table = algo.tail(20)
    table.to_json('../web/templates/table.json', orient='records')
    a = Algo.plot_percentiles(algo)
    b = Algo.plot_positionR(algo)
    c = Algo.market_returnsR(algo)
    d = Algo.nineM(algo)
    e = Algo.plot_positionT(algo)
    f = Algo.plot_gainzT(algo)
    g = Algo.regime(algo)
    h = Algo.folio(algo)
    i = Algo.folioDB()
    j = Algo.dualPlot(algo)
    time.sleep(60)