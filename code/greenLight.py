import pandas as pd
import numpy as np
import time
import datetime
import requests
import matplotlib.pyplot as plt
from client import engine
plt.rcParams['figure.figsize'] = [16.0, 6.0]



class greenLight():

    def __init__(self):
        self.self = self


    def fullstate(self, df) -> pd.DataFrame:
        """ Trend following algo"""
        short_ma = 9
        long_ma = 21
        self['9-min'] = self['close'].rolling(short_ma).mean().shift()
        self['21-min'] = self['close'].rolling(long_ma).mean().shift()
        self['signal'] = np.where(self['9-min'] > self['21-min'], 1, np.nan)
        self['signal'] = np.where(
            self['9-min'] < self['21-min'], -1, self['signal'])
        self.dropna(inplace=True)
        self['market_returns'] = np.log(self['close']).diff()
        self['trend_returns'] = self['signal'] * self['market_returns'].shift()
        self['entry'] = self.signal.diff()
        self["ma"] = self['close'].rolling(9).mean().shift()
        self['ratio'] = self['close'] / self['ma']
        percentiles = [5, 10, 50, 90, 95]
        p = np.percentile(self['ratio'].dropna(), percentiles)
        short = p[-1]
        long = p[0]
        self['position'] = np.where(self.ratio >= short, -1, np.nan)
        self['position'] = np.where(self.ratio < long, 1, self['position'])
        self['position'] = self['position'].ffill()
        self['entryR'] = self.position.diff()
        self['range_returns'] = self['market_returns'] * \
            self['position'].shift()
        self['sign'] = np.where(self['trend_returns'] >
                                self['range_returns'], 1, np.nan)
        self['sign'] = np.where(self['trend_returns'] <
                                self['range_returns'], -1, self['sign'])
        return self

    def folio(self) -> pd.DataFrame:
        self['just_date'] = self['time'].dt.date
        #column for negative and positive
        self=self.dropna()
        self['rangeSign'] = np.where(self['range_returns'] < 0, 'neg','pos')
        self['trendSign'] = np.where(self['trend_returns'] < 0, 'neg','pos')

        #consecutive groups
        self['rangeSeries'] = self['rangeSign'].ne(self['rangeSign'].shift()).cumsum()
        self['trendSeries'] = self['trendSign'].ne(self['trendSign'].shift()).cumsum()

        #removed groups with length more like 2
        df = self[self['rangeSeries'].map(self['rangeSeries'].value_counts()).gt(2)]
        df = self[self['trendSeries'].map(self['trendSeries'].value_counts()).gt(2)]

        #tested if order `pos-neg` of groups, if not removed groups
        m1 = df['rangeSign'].eq('pos') & df['rangeSign'].shift(-1).eq('neg')
        m2 = df['rangeSign'].eq('neg') & df['rangeSign'].shift().eq('pos')
        m3 = df['trendSign'].eq('pos') & df['trendSign'].shift(-1).eq('neg')
        m4 = df['trendSign'].eq('neg') & df['trendSign'].shift().eq('pos')
        groupsR = df.loc[m1 | m2, 'rangeSeries']
        df = df[df['rangeSeries'].isin(groupsR)].copy()
        df['rangePairs'] = (df['rangeSign'].ne(df['rangeSign'].shift()) & df['rangeSign'].eq('pos')).cumsum()
        groupsT = df.loc[m3 | m4, 'trendSeries']
        df = df[df['trendSeries'].isin(groupsT)].copy()
        df['trendPairs'] = (df['trendSign'].ne(df['trendSign'].shift()) & df['trendSign'].eq('pos')).cumsum()
        rangeTradeCounts = df['rangeSeries'].nunique()
        trendTradeCounts = df['trendSeries'].nunique()
        totalTrades = rangeTradeCounts + trendTradeCounts
        return df

    
    def trendPositions(self):
        plt.figure(figsize=(30, 10))
        fig = plt.figure(facecolor=(1, 1, 1))
        y = self.iloc[-500:]['time']
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(self.iloc[-500:]['close'], label = 'BTC')
        plt.plot(self.iloc[-500:]['9-min'], label = '9-min')
        plt.plot(self.iloc[-500:]['21-min'], label = '21-min')
        plt.plot(self[-500:].loc[self.entry == 2].index, self[-500:]['9-min'][self.entry == 2], "^",
                color = "g", markersize = 12, label= "Long")
        plt.plot(self[-500:].loc[self.entry == -2].index, self[-500:]['21-min'][self.entry == -2], "v",
                color = "r", markersize = 12, label="Short")
        plt.legend(loc=2);
        plt.savefig('../web/assets/trendPositions.png')

    def trendRets(self):
        plt.figure(figsize=(30, 10))
        fig = plt.figure(facecolor=(1, 1, 1))
        y=self.iloc[-500:]['time']
        self['trend_returns'] = self.signal * self.market_returns
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(np.exp(self.iloc[-500:]['market_returns']).cumprod(),label = "Buy/Hold")
        plt.plot(np.exp(self.iloc[-500:]['trend_returns']).cumprod(),label = "Strat")
        plt.legend()
        plt.savefig('../web/assets/trendRets.png')

    def assetLineChart(self):
        plt.figure(figsize=(30, 10))
        fig = plt.figure(facecolor=(1, 1, 1))
        y=self.iloc[-500:]['time']
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(self.iloc[-500:]['close'], label = 'BTC')
        plt.plot(self.iloc[-500:]['9-min'], label = '9-min')
        plt.legend(loc=2)
        plt.savefig('../web/assets/btc1m9ma.png')

    def rangePositions(self):
        plt.figure(figsize=(30, 10))
        fig = plt.figure(facecolor=(1, 1, 1))
        y=self.iloc[-500:]['time']
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(self.iloc[-500:]['position'].dropna())
        plt.savefig('../web/assets/rangeStatus.png')

    def plot_percentiles(self):
        plt.figure(figsize=(30, 10))
        """ Plots price percenitles"""
        sb.set()
        y = self.iloc[-500:]['time']
        percentiles = [5, 10, 50, 90, 95]
        p = np.percentile(self['ratio'].dropna(), percentiles)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(self.iloc[-500:]['ratio'].dropna())
        plt.axhline(p[0], c=(.5, .5, .5), ls='--')
        plt.axhline(p[2], c=(.5, .5, .5), ls='--')
        plt.axhline(p[-1], c=(.5, .5, .5), ls='--')
        plt.savefig('../web/assets/rangePercentiles.png')

    def rangeRets(self):
        plt.figure(figsize=(30, 10))
        y=self.iloc[-500:]['time'].dropna()
        fig = plt.figure(facecolor=(1, 1, 1))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(np.exp(self.iloc[-500:]['market_returns'].dropna()).cumprod(), label='Buy/Hold')
        plt.plot(np.exp(self.iloc[-500:]['range_returns'].dropna()).cumprod(), label='Strategy')
        plt.legend();
        plt.savefig('../web/assets/rangeRets.png')

    def dualPlot(self):
        plt.figure(figsize=(30, 10))
        y=self.iloc[-500:]['time']
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(self.iloc[-500:]['close'], label = 'BTC')
        plt.plot(self.iloc[-500:]['9-min'], label = '9-min')
        plt.plot(self[-500:].loc[self.entryR == 2].index, self[-500:]['9-min'][self.entryR == 2], "^",
                color = "r", markersize = 12, label= "Short")
        plt.plot(self[-500:].loc[self.entryR == -2].index, self[-500:]['9-min'][self.entryR == -2], "v",
                color = "g", markersize = 12, label="Long")
        plt.legend(loc=2);
        plt.savefig('../web/assets/dualPlot.png')

    def table1(self):
        table = self.tail(20)
        table.to_json('../web/templates/table.json', orient='records')

