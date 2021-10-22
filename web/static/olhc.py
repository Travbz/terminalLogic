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



socket = 'wss://ftx.com/ws/'
api_key = '13tLs18EEiz6pYp4QD77dM0mkctM1jWPbs1THoDv'
secret_key = '0qDUuVo59mKew5-v9jcY9Kb79wYbHE-TMk9nf85k'
minutes_processed = {}
minute_candlesticks = []
current_tick = None
previous_tick = None 

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
    global current_tick, previous_tick
    previous_tick = current_tick
    current_tick = json.loads(message)
    # print(current_tick['data'])
    tick_datetime_object = datetime.fromtimestamp(current_tick['data']['time']).strftime("%m/%d/%Y %H:%M:%S")
    tick_datetime = dateutil.parser.parse(tick_datetime_object)
    tick_dt = tick_datetime.strftime("%m/%d/%Y %H:%M")
    tick_min = tick_datetime.strftime("%M")
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
    if len(minute_candlesticks) > 9:
        current_candlestick['9ma'] = current_candlestick['close'].rolling(9).mean()
    
    print('==Candlesticks==')
    for candlestick in minute_candlesticks:
        print(candlestick)

def on_error(ws,error):
    print(error)

ws = websocket.WebSocketApp(socket,on_open=on_open,on_close=on_close,on_message=on_message,on_error=on_error)

ws.run_forever()
