import pandas as pd
import numpy as np
import time
import seaborn as sb
import requests
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
import matplotlib.pyplot as plt
from IPython import get_ipython
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
from ciso8601 import parse_datetime
from requests import Request, Session, Response
import sys
import hmac
import urllib.parse
import datetime
import sqlite3
import csv

class Algo():

    def __init__(self, df):
        self.self = self
        self.df = df

    def ranger(self, df) -> pd.DataFrame:
        """ Basic Indicator values inserted into df, mostly for the range algo"""
        self["ma"] = self['close'].rolling(9).mean()
        self['ratio'] = self['close'] / self['ma']
        percentiles = [5, 10, 50, 90, 95]
        p = np.percentile(self['ratio'].dropna(), percentiles)
        short = p[-1]
        long = p[0]
        self['position'] = np.where(self.ratio >= short, -1, np.nan)
        self['position'] = np.where(self.ratio < long, 1, self['position'])
        self['position'] = self['position'].ffill()
        self['returnsR'] = np.log(self["close"]).diff()
        self['strat_returnR'] = self['returnsR'] * self['position'].shift()
        return self

    def plot_R(self):
        plt.rcParams['figure.figsize'] = [16.0, 6.0]
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

    def plot_positionR(self):
        plt.rcParams['figure.figsize'] = [16.0, 6.0]

        fig = plt.figure(facecolor=(1, 1, 1))
        y=self.iloc[-500:]['time']
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(self.iloc[-500:]['position'].dropna())
        plt.savefig('../web/assets/rangeStatus.png')

    def market_returnsR(self):
        plt.rcParams['figure.figsize'] = [16.0, 6.0]

        fig = plt.figure(facecolor=(1, 1, 1))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(
            np.exp(self.iloc[-500:]['market_returns'].dropna()).cumprod(), label='Buy/Hold')
        plt.plot(
            np.exp(self.iloc[-500:]['range_returns'].dropna()).cumprod(), label='Strategy')
        plt.xticks(rotation=90)
        plt.legend()
        plt.savefig('../web/assets/rangeRets.png')


    def nineM(self):
        plt.rcParams['figure.figsize'] = [16.0, 6.0]
        fig = plt.figure(facecolor=(1, 1, 1))
        y=self.iloc[-500:]['time']
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(self.iloc[-500:]['close'], label = 'BTC')
        plt.plot(self.iloc[-500:]['9-min'], label = '9-min')
        plt.legend(loc=2);
        plt.savefig('../web/assets/btc1m9ma.png')

    def range_gainz(self):

        print("Market Returns: ", np.exp(
            self.market_returns).cumprod().iloc[-1])
        print("Range Strategy Returns: ", np.exp(
            self.range_returns).cumprod().iloc[-1])

    def trendy(self, df) -> pd.DataFrame:
        """ Trend following algo"""
        short_ma = 9
        long_ma = 21
        self['9-min'] = self['close'].rolling(short_ma).mean().shift()
        self['21-min'] = self['close'].rolling(long_ma).mean().shift()
        self['signal'] = np.where(self['9-min'] > self['21-min'], 1, np.nan)
        self['signal'] = np.where(
            self['9-min'] < self['21-min'], -1, self['signal'])
        self.dropna(inplace=True)
        self['returnsT'] = np.log(self['close']).diff()
        self['strat_returnT'] = self['signal'] * self['returnsT'].shift()
        self['entry'] = self.signal.diff()
        return self

    def plot_positionT(self):
        plt.rcParams['figure.figsize'] = [16.0, 6.0]
        """Plots long short flips on line chart for trend algo 'T' """
        plt.rcParams['figure.figsize'] = 30, 10
        plt.grid(True, alpha=.3)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        y = self.iloc[-500:]['time']
        plt.plot(self.iloc[-500:]['close'], label='BTC')
        plt.plot(self.iloc[-500:]['9-min'], label='9-min')
        plt.plot(self.iloc[-500:]['21-min'], label='21-min')
        plt.plot(self[-500:].loc[self.entry == 2].index, self[-500:]['9-min'][self.entry == 2], "^",
                 color="g", markersize=12, label="Long")
        plt.plot(self[-500:].loc[self.entry == -2].index, self[-500:]['21-min'][self.entry == -2], "v",
                 color="r", markersize=12, label="Short")
        plt.legend(loc=2)
        plt.savefig('../web/assets/trendPositions1.png')

    def plot_gainzT(self):
        plt.rcParams['figure.figsize'] = [16.0, 6.0]
        fig = plt.figure(facecolor=(1, 1, 1))
        y=self.iloc[-500:]['time']
        self['trend_returns'] = self.signal * self.market_returns
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.plot(np.exp(self.iloc[-500:]['market_returns']).cumprod(),label = "Buy/Hold")
        plt.plot(np.exp(self.iloc[-500:]['trend_returns']).cumprod(),label = "Strat")
        plt.legend()
        plt.savefig('../web/assets/trendRets.png')

    def trend_gainz(self):

        print("Market Returns: ", np.exp(
            self.market_returns).cumprod().iloc[-1])
        print("Trend Strategy Returns: ", np.exp(
            self.trend_returns).cumprod().iloc[-1])

    def dualPlot(self):
        fig = plt.figure(facecolor=(1, 1, 1))
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





    def Z_scoreR(self, df) -> pd.DataFrame:
        '''Args, needs col 'strat_return with +/- returns'''
        # column for negative and positive
        df.dropna().head(100)
        df['posneg'] = np.where(df['strat_returnR'] < 0, 'neg', 'pos')
        # consecutive groups
        df['series'] = df['posneg'].ne(df['posneg'].shift()).cumsum()
        # removed groups with length more like 2
        df = df[df['series'].map(df['series'].value_counts()).gt(2)]
        # tested if order `pos-neg` of groups, if not removed groups
        m1 = df['posneg'].eq('pos') & df['posneg'].shift(-1).eq('neg')
        m2 = df['posneg'].eq('neg') & df['posneg'].shift().eq('pos')
        groups = df.loc[m1 | m2, 'series']
        df = df[df['series'].isin(groups)].copy()
        df['pairs'] = (df['posneg'].ne(df['posneg'].shift())
                       & df['posneg'].eq('pos')).cumsum()

        N = len(df['series'].dropna())
        R = df['series'].dropna().nunique()
        W = len(df.loc[df.strat_returnR > 0])
        L = len(df.loc[df.strat_returnR < 0])
        P = 2*W*L

        Z_score = (N*(R-0.5)-P)/((P*(P-N))/(N-1))**(1/2)

        return float(Z_score)

    def Z_scoreT(self, df) -> pd.DataFrame:
        '''Args, needs col 'strat_return with +/- returns'''
        # column for negative and positive
        self['posneg'] = np.where(self['trend_returns'] < 0, 'neg', 'pos')
        # consecutive groups
        self['series'] = self['posneg'].ne(self['posneg'].shift()).cumsum()
        # removed groups with length more like 2
        self = self[self['series'].map(self['series'].value_counts()).gt(2)]
        # tested if order `pos-neg` of groups, if not removed groups
        m1 = self['posneg'].eq('pos') & self['posneg'].shift(-1).eq('neg')
        m2 = self['posneg'].eq('neg') & self['posneg'].shift().eq('pos')
        groups = self.loc[m1 | m2, 'series']
        self = self[self['series'].isin(groups)].copy()
        self['pairs'] = (self['posneg'].ne(self['posneg'].shift())
                       & self['posneg'].eq('pos')).cumsum()

        N = len(self['series'].dropna())
        R = self['series'].dropna().nunique()
        W = len(self.loc[self.trend_returns> 0])
        L = len(self.loc[self.trend_returns < 0])
        P = 2*W*L

        Z_score = (N*(R-0.5)-P)/((P*(P-N))/(N-1))**(1/2)

        return float(Z_score)

    def marketState(d, b):
        ''' Func to determine market state, and shifts trading engine to that regime'''
        prev_state = []
        prev_state_name = []
        state_selector_switch = []
        d = float(d)
        b = float(b)
        if d < 0 and d < b or d > 0 and d > b:
            prev_state.append(d)
            prev_state_name.append("Trend")
            state_selector_switch.append(1)  # pos 1 indicates trend conditions
            print(
                "Trend detector algo's ZScore indicates there is a 'Trending' market state..", "..Z Score:", d)
            print('Here is the trend_df with our trend indicator suite')
            print(
                'All trades for the next period will be placed according to the conditions within...')
            return state_selector_switch
        elif b < 0 and b < d or b > 0 and b > d:
            prev_state.append(b)
            prev_state_name.append("Range")
            # neg -1 indicates range conditions
            state_selector_switch.append(-1)
            print(
                "Range detector algo's ZScore indicates there is a 'Ranging' market state..", "..Z Score:", b)
            print('Here is the range_df with our range indicator suite')
            print(
                'All trades for the next period will be placed according to the conditions within...')
            return state_selector_switch
        else:
            return state_selector_switch
            print(f"Market state detector algo's have exact same values, previous state algo will be used for the next period..")
            print(
                f" Previous State: {prev_state_name}, previous Z_score: {prev_state}, Current Z_score tie: {b}")

    def stateTest(df):
        """ Calls FTX REST APi, gathers data, flips rows, adds TA for algo's, and calcs Z_score's' for each algo"""
        df = df.dropna()
        df = df[::-1]
        trend_df = Algo.trendy(df, df)
        range_df = Algo.ranger(df, df)
        a = Algo(range_df)
        b = a.Z_scoreR(range_df)
        c = Algo(trend_df)
        d = c.Z_scoreT(trend_df)
        return d, b

    def stateSelector(df, int):

        df = df.dropna()
        '''func to switch between mean reversion and trend phase'''
        if int == -1:
            return Algo.ranger(df, df)
        elif int == 1:
            return Algo.trendy(df, df)

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
        df['just_date'] = df['time'].dt.date
        df['just_date']
        # Set initial capital
        initial_capital = float(22000.0)
        # Create df positions
        positions = pd.DataFrame(index=df.time.index).fillna(0.0)
        # Buy 2 BTC
        positions['BTCPERP'] = 1*df['signal']
        # Initilize portfolio w value owned
        portfolio = positions.multiply(df['close'], axis=0)
        # Store diff in shares owned
        pos_diff = positions.diff()
        # Add 'holdings' to portfolio
        portfolio['holdings'] = (positions.multiply(df['close'], axis=0)).sum(axis=1)
        # Add 'cash' to portfolio
        portfolio['cash'] = initial_capital - (pos_diff.multiply(df['close'], axis=0)).sum(axis=1).cumsum()
        # Add 'total' to portfolio
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        # Add 'returns' to portfolio
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['time'] = df['time']
        p = portfolio[-1:]
        p.drop(columns=['time'], inplace=True)
        p = p.reset_index(drop=True)
        p.to_json('../web/templates/portfolio2.json', orient='records')
        fig = plt.figure(facecolor=(1, 1, 1))
        x = portfolio.iloc[-200:]['time']
        y = portfolio.iloc[-200:]['total']
        plt.xticks(fontsize=22, color="black", rotation=25)
        plt.xlabel('Time', color='black',fontsize=22)
        plt.yticks(fontsize=22, color='black')
        plt.ylabel('Value', color='black',fontsize=22)
        plt.locator_params(axis='x', nbins=8)
        plt.plot(x,y)
        plt.savefig('../web/assets/portfolioStandings.png')
        plt.show()
        portfolio.to_csv("../web/assets/portfolio.csv", index=False)
    
    
        

    def folioDB():
        conn = sqlite3.connect('folio.db')
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS folio (BTCPERP int, holdings int, cash int, total int, returns int, time text)''')
        folioTable = pd.read_csv('../web/assets/portfolio.csv') # load to DataFrame
        folioTable.to_sql('orders', conn, if_exists='append', index = False) # write to sqlite table





    def regime(df):    
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        steps = [('imputation', imp),
                ('scaler',StandardScaler()),
                ('lasso',Lasso())]        

        pipeline =Pipeline(steps)


        parameters = {'lasso__alpha':np.arange(0.0001,10,.0001),
                    'lasso__max_iter':np.random.uniform(100,100000,4)}
        from pandas_datareader import data as web
        from sklearn import mixture as mix
        import seaborn as sns 
        import matplotlib.pyplot as plt
        reg = rcv(pipeline, parameters,cv=5)
        X=df[['open','high','low','close']]
        y =df['close']
        avg_err={}
        avg_err={}
        avg_train_err = {}
        for t in np.arange(50,97,3):
            get_ipython().magic('reset_selective -f reg1')
            split = int(t*len(X)/100)
            reg.fit(X[:split],y[:split])
            best_alpha = reg.best_params_['lasso__alpha']
            best_iter = reg.best_params_['lasso__max_iter']
            reg1 = Lasso(alpha=best_alpha,max_iter=best_iter)
            X = imp.fit_transform(X,y)
            reg1.fit(X[:split],y[:split])
            df['P_C_%i'%t] = 0.
            df.iloc[:,df.columns.get_loc('P_C_%i'%t)] = reg1.predict(X[:])
            df['Error_%i'%t] = np.abs(df['P_C_%i'%t]-df['close'])
            
            e = np.mean(df['Error_%i'%t][split:])
            train_e = np.mean(df['Error_%i'%t][:split])
            avg_err[t] = e
            avg_train_err[t] = train_e

        plt.rcParams['figure.figsize'] = [4.0, 4.0]
        fig = plt.figure(facecolor=(1, 1, 1))
        Range =df['high'][split:]-df['low'][split:]
        plt.scatter(list(avg_train_err.keys()),list(avg_train_err.values()),label='train_error')
        plt.legend(loc='best')
        avgRange = np.average(Range)
        plt.title(f'Avg Range = %1.2f'%avgRange)
        plt.savefig('../web/assets/lasso-error.png')

        plt.rcParams['figure.figsize'] = [4.0, 4.0]
        fig = plt.figure(facecolor=(1, 1, 1))
        Range =df['high'][split:]-df['low'][split:]
        # ------------------------------------------------------------------------ added code below.
        plt.scatter(list(avg_err.keys()),list(avg_err.values()), label='test_error')
        # ---------------------------------------------------------------------------
        plt.scatter(list(avg_train_err.keys()),list(avg_train_err.values()),label='train_error')
        plt.legend(loc='best')
        avR = np.average(Range)
        plt.title(f'Avg Range = %1.2f'%avR)
        plt.savefig('../web/assets/train-test-error.png')

        df=df[['open','high','low','close']]
        df['open']=df['open'].shift(1)
        df['high']=df['high'].shift(1)
        df['low']=df['low'].shift(1)
        df['close']=df['close'].shift(1)
        df=df[['open','high','low','close']]
        df=df.dropna()
        unsup = mix.GaussianMixture(n_components=3, 
                                    covariance_type="spherical", 
                                    n_init=100, 
                                    random_state=42)
        unsup.fit(np.reshape(df,(-1,df.shape[1])))
        regime = unsup.predict(np.reshape(df,(-1,df.shape[1])))
        df['Return']= np.log(df['close']/df['close'].shift(1))
        Regimes=pd.DataFrame(regime,columns=['Regime'],index=df.index)\
                            .join(df, how='inner')\
                                .assign(market_cu_return=df.Return.cumsum())\
                                        .reset_index(drop=False)\
                                                    .rename(columns={'index':'Date'})
        plt.rcParams['figure.figsize'] = [16.0, 6.0]
        order=[0,1,2]
        fig = sns.FacetGrid(data=Regimes,hue='Regime',hue_order=order,aspect=2,height= 4)
        fig.map(plt.scatter,'Date','market_cu_return', s=4).add_legend(labelcolor='white')
        plt.tick_params(colors='white', grid_color='black')
        plt.rcParams['text.color']='w'

        plt.grid()
        plt.savefig('../web/assets/lasso.png', bbox_inches='tight')
        plt.show()
        for i in order:
            print('Mean for regime %i: '%i,unsup.means_[i][0])
            print('Co-Variancefor regime %i: '%i,(unsup.covariances_[i]))
