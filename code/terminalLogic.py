import pandas as pd
import numpy as np
import time
import requests
import matplotlib.pyplot as plt


class Algo():

  def __init__(self, df):
     self.self = self
     self.df = df

  def ranger(self,df)->pd.DataFrame:

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

  def plot_percentiles(self):
    """ Plots price percenitles"""
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set()
    percentiles = [5, 10, 50, 90, 95]
    p = np.percentile(self['ratio'].dropna(), percentiles)
    self['ratio'].dropna().plot(legend = True)
    plt.axhline(p[0], c= (.5,.5,.5), ls='--')
    plt.axhline(p[2], c= (.5,.5,.5), ls='--')
    plt.axhline(p[-1], c= (.5,.5,.5), ls='--');
    plt.savefig('../web/assets/rangePercentiles.png')

  def plot_positionR(self):
    """ Plots positions takens for range algo """
    self.position.dropna().plot()
    plt.savefig('../web/assets/rangePositions.png')


  def market_returnsR(self):
    """ Plots returns for the range algo df thus returns 'R' """
    plt.plot(np.exp(self['market_returns'].dropna()).cumprod(), label='Buy/Hold')
    plt.plot(np.exp(self['range_returns'].dropna()).cumprod(), label='Strategy')
    plt.xticks(rotation=90)
    plt.legend();
    plt.savefig('../web/assets/rangeRets.png')


  def range_gainz(self):

    print("Market Returns: ", np.exp(self.market_returns).cumprod().iloc[-1])
    print("Range Strategy Returns: ", np.exp(self.range_returns).cumprod().iloc[-1])
  
  def trendy(self,df)->pd.DataFrame:

    """ Trend following algo"""
    short_ma = 9
    long_ma = 21
    self['9-min'] = self['close'].rolling(short_ma).mean().shift()
    self['21-min'] = self['close'].rolling(long_ma).mean().shift()
    self['signal'] = np.where(self['9-min'] > self['21-min'], 1, np.nan)
    self['signal'] = np.where(self['9-min'] < self['21-min'], -1, self['signal'])
    self.dropna(inplace=True)
    self['returnsT'] = np.log(self['close']).diff()
    self['strat_returnT'] = self['signal'] * self['returnsT'].shift()
    self['entry'] = self.signal.diff()
    return self
    
  def plot_positionT(self):
    """Plots long short flips on line chart for trend algo 'T' """
    plt.rcParams['figure.figsize'] = 30,10
    plt.grid(True, alpha = .3)
    plt.plot(self.iloc[-500:]['close'], label = 'BTC')
    plt.plot(self.iloc[-500:]['9-min'], label = '9-min')
    plt.plot(self.iloc[-500:]['21-min'], label = '21-min')
    plt.plot(self[-500:].loc[self.entry == 2].index, self[-500:]['9-min'][self.entry == 2], "^",
            color = "g", markersize = 12, label= "Long")
    plt.plot(self[-500:].loc[self.entry == -2].index, self[-500:]['21-min'][self.entry == -2], "v",
            color = "r", markersize = 12, label="Short")
    plt.legend(loc=2);

  def plot_gainzT(self):
    self['trend_returns'] = self.signal * self.market_returns
    plt.plot(np.exp(self.market_returns).cumprod(),label = "Buy/Hold")
    plt.plot(np.exp(self.trend_returns).cumprod(),label = "Strat")
    plt.legend()
    plt.savefig('../web/assets/trendRets.png')

  def trend_gainz(self):

    print("Market Returns: ", np.exp(self.market_returns).cumprod().iloc[-1])
    print("Trend Strategy Returns: ", np.exp(self.trend_returns).cumprod().iloc[-1])


  def Z_scoreR(self, df)->pd.DataFrame:
    '''Args, needs col 'strat_return with +/- returns'''
    #column for negative and positive
    df.dropna().head(100)
    df['posneg'] = np.where(df['strat_returnR'] < 0, 'neg','pos')
    #consecutive groups
    df['series'] = df['posneg'].ne(df['posneg'].shift()).cumsum()
    #removed groups with length more like 2
    df = df[df['series'].map(df['series'].value_counts()).gt(2)]
    #tested if order `pos-neg` of groups, if not removed groups
    m1 = df['posneg'].eq('pos') & df['posneg'].shift(-1).eq('neg')
    m2 = df['posneg'].eq('neg') & df['posneg'].shift().eq('pos')
    groups = df.loc[m1 | m2, 'series']
    df = df[df['series'].isin(groups)].copy()
    df['pairs'] = (df['posneg'].ne(df['posneg'].shift()) & df['posneg'].eq('pos')).cumsum()

    N = len(df['series'].dropna())
    R = df['series'].dropna().nunique()
    W = len(df.loc[df.strat_returnR> 0])
    L = len(df.loc[df.strat_returnR< 0])
    P = 2*W*L

    Z_score = (N*(R-0.5)-P)/((P*(P-N))/(N-1))**(1/2)
    
    return float(Z_score)


  def Z_scoreT(self, df)->pd.DataFrame:
      '''Args, needs col 'strat_return with +/- returns'''
      #column for negative and positive
      df.dropna().head(100)
      df['posneg'] = np.where(df['strat_returnT'] < 0, 'neg','pos')
      #consecutive groups
      df['series'] = df['posneg'].ne(df['posneg'].shift()).cumsum()
      #removed groups with length more like 2
      df = df[df['series'].map(df['series'].value_counts()).gt(2)]
      #tested if order `pos-neg` of groups, if not removed groups
      m1 = df['posneg'].eq('pos') & df['posneg'].shift(-1).eq('neg')
      m2 = df['posneg'].eq('neg') & df['posneg'].shift().eq('pos')
      groups = df.loc[m1 | m2, 'series']
      df = df[df['series'].isin(groups)].copy()
      df['pairs'] = (df['posneg'].ne(df['posneg'].shift()) & df['posneg'].eq('pos')).cumsum()

      N = len(df['series'].dropna())
      R = df['series'].dropna().nunique()
      W = len(df.loc[df.strat_returnT> 0])
      L = len(df.loc[df.strat_returnT< 0])
      P = 2*W*L

      Z_score = (N*(R-0.5)-P)/((P*(P-N))/(N-1))**(1/2)
      
      return float(Z_score)

  def marketState(d,b):    
      ''' Func to determine market state, and shifts trading engine to that regime'''
      prev_state = []
      prev_state_name = []
      state_selector_switch = []
      d = float(d)
      b = float(b)
      if d < 0 and d < b or d > 0 and d > b:
          prev_state.append(d)
          prev_state_name.append("Trend")
          state_selector_switch.append(1) # pos 1 indicates trend conditions       
          print("Trend detector algo's ZScore indicates there is a 'Trending' market state..", "..Z Score:", d)
          print('Here is the trend_df with our trend indicator suite')
          print('All trades for the next period will be placed according to the conditions within...')
          return state_selector_switch   
      elif b < 0 and b < d or b > 0 and b > d:
          prev_state.append(b)
          prev_state_name.append("Range")
          state_selector_switch.append(-1) # neg -1 indicates range conditions
          print("Range detector algo's ZScore indicates there is a 'Ranging' market state..", "..Z Score:", b)
          print('Here is the range_df with our range indicator suite')
          print('All trades for the next period will be placed according to the conditions within...')
          return state_selector_switch
      else:
          return state_selector_switch
          print(f"Market state detector algo's have exact same values, previous state algo will be used for the next period..")
          print(f" Previous State: {prev_state_name}, previous Z_score: {prev_state}, Current Z_score tie: {b}")




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
      return d,b


  def stateSelector(df, int):
    
      df = df.dropna()
      '''func to switch between mean reversion and trend phase'''
      if int == -1:
          return  Algo.ranger(df,df)
      elif int == 1:
          return Algo.trendy(df,df)




  def fullstate(self,df)->pd.DataFrame:

      """ Trend following algo"""
      short_ma = 9
      long_ma = 21
      self['9-min'] = self['close'].rolling(short_ma).mean().shift()
      self['21-min'] = self['close'].rolling(long_ma).mean().shift()
      self['signal'] = np.where(self['9-min'] > self['21-min'], 1, np.nan)
      self['signal'] = np.where(self['9-min'] < self['21-min'], -1, self['signal'])
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
      self['range_returns'] = self['market_returns'] * self['position'].shift()
      self['sign'] = np.where(self['trend_returns'] > self['range_returns'], 1, np.nan)
      self['sign'] = np.where(self['trend_returns'] < self['range_returns'], -1, self['sign'])
      return self


