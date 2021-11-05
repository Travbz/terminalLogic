import websocket
import pandas as pd 
import numpy as np 
import json 
import hmac 
import time 
import matplotlib.pyplot as plt
from datetime import datetime,  timedelta
import threading
import dateutil.parser
import sqlite3

# class MaxSizeList(list):
#     def __init__(self, maxlen):
#         self._maxlen = maxlen

#     def append(self, element):
#         self.__delitem__(slice(0, len(self) == self._maxlen))
#         super(MaxSizeList, self).append(element)
# # example: a = MaxSizeList(3)


socket = 'wss://ftx.com/ws/'
api_key = '13tLs18EEiz6pYp4QD77dM0mkctM1jWPbs1THoDv'
secret_key = '0qDUuVo59mKew5-v9jcY9Kb79wYbHE-TMk9nf85k'
minutes_processed = {}
minute_candlesticks =[]
current_tick = None
previous_tick = None 
connection = sqlite3.connect('ohlc.db')
cursor = connection.cursor()
time_done = []
try:
    cursor.execute('''CREATE TABLE IF NOT EXISTS btc1m
                            (startTime text, o real, high real, low real, close real)''')
    # cursor.executemany("INSERT OR IGNORE INTO btc1m VALUES (?,?,?,?)", startTime, open1m, high1m, low1m, close1m)
except Exception as e:
    connection.commit()

def on_open(ws):
    print('connected')
    ts = int(time.time() * 1000)
    signa = hmac.new(secret_key.encode(), f'{ts}websocket_login'.encode(), 'sha256').hexdigest()
    auth = {'op': 'login', 'args': {'key': api_key,
                                'sign': signa, 
                                'time': ts}}
    ws.send(json.dumps(auth))
    data = {'op': 'subscribe', 'channel': 'ticker', 'market': 'BTC-PERP'}
    ws.send(json.dumps(data))

def on_close(ws):
    print('disconnected')

def on_message(ws,message):
    global current_tick, previous_tick, minute_candlesticks, time_done
    print("NEW MESSAGE RECEIVED", time_done)
    previous_tick = current_tick
    current_tick = json.loads(message)
    # print(current_tick['data'])
    tick_datetime_object = datetime.fromtimestamp(current_tick['data']['time']).strftime("%m/%d/%Y %H:%M:%S")
    tick_datetime = dateutil.parser.parse(tick_datetime_object)
    tick_dt = tick_datetime.strftime("%m/%d/%Y %H:%M")
    tick_min = tick_datetime.strftime("%M")
    print(tick_dt)
    if not tick_dt in minutes_processed:
        minutes_processed[tick_dt] = True
        if len(minute_candlesticks) > 0:
            minute_candlesticks[-1]['close'] = previous_tick['data']['last']
        minute_candlesticks.append({
            "startTime": tick_dt,
            "open": current_tick['data']['last'],
            "high": current_tick['data']['last'],
            "low": current_tick['data']['last'],
        })
    if len(minute_candlesticks) > 0:
        current_candlestick = minute_candlesticks[-1]
        if current_tick['data']['last'] > current_candlestick['high']:
            current_candlestick['high'] = current_tick['data']['last']
        if current_tick['data']['last'] < current_candlestick['low']:
            current_candlestick['low'] = current_tick['data']['last']
        print('==Candlesticks==')
        candle_list = []
        for candle in minute_candlesticks:
            startTime = candle['startTime']
            o = candle['open']
            high = candle['high']
            low = candle['low']
            close = candle['close']
            #candle_list.append(startTime['startTime'], o['open'], high['high'], low['low'], close['close'])
            print(candle)

            if startTime not in time_done:
                print("NEW VALUE", startTime)
                try:
                    print("executing query")
                    cursor.execute("INSERT INTO btc1m VALUES (?,?,?,?,?)", (startTime, o, high, low, close) )
                    connection.commit()
                except Exception as e:
                    print(e)
                time_done.append(startTime)
        
        #print(startTime, o, high, low, close)
        #{'startTime': '10/25/2021 11:50', 'open': 63281.0, 'high': 63320.0, 'low': 63103.0, 'close': 63104.0}
        #{ '}
        #"INSERT OR IGNORE INTO btc1m VALUES ('10/25/2021 11:50' ,?,?,?,?)",

        try:
            print("executing query")
            cursor.execute("INSERT INTO btc1m VALUES (?,?,?,?,?)", (startTime, o, high, low, close) )
            connection.commit()
        except Exception as e:
            print(e)
            #connection.commit()

def on_error(ws,error):
    print(error)

ws = websocket.WebSocketApp(socket,on_open=on_open,on_close=on_close,on_message=on_message,on_error=on_error)

ws.run_forever()
