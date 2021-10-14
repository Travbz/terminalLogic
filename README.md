# terminalLogic

## terminalLogic aims to eliminate discretion from the users trading strategy and provides machine learning tools to optimize parameters for its trading algorythms in an attempt to be profitable.


### Basic Concept - Markets spend X_amount of time in a range or mean reverting state and X_amount of time in a trending state. terminalLogic will use ML to determine which state the market is in and will deploy a trading strategy that is suitable for that state. There are two different ML stacks being built, each with 2 algos in it that generate a Zscore based on a backtest of the last set time period. The algo with the highest Zscore will then be used for the next set time period. Once the market_state has been determined and the appropriate algo deployed, signals are sent to a script that does some basic calculations to output a suggested position size based on num_of_open_positions, account balance, risk exposure tolerances, and a double check to see if latency has caused the edge to go stale. If everything checks out an order manager places the order on FTX Exchange. All trades, market conditions and changes in portfolio are logged in a db and plotted for interpretation. 

### Due to poor internet connections, possible outages, and limited processing power in my home laptop.. terminalLogic will be hosted on a 300GB AWS Linux machine complete with every tool it needs to run 24/7. I chose a VPS because the ML stacks are very heavy and require a lot of horsepower to solve quickly and the historical market data needs to be accessed 100% of the time, uninterupted. 

### Really this repo is incomplete as i have 2 local reops with all of the in-progress components in it. I am using github soley as a backup. Much of this exists soley in colab and SageMaker.

## Architecture, estimated 50% complete atm. Most functions for API interaction with the exchange are working, SQL database works, ML algos/functions to identify market state work, I can poll data, backtest results on historical market data, backtest portfolio performance etc... I need to work on order placement/management, risk management, and incorporating risk management into portfolio performance/backtesting so i can adjust some ML algo parameters and then its really just job scheduling. After that i should be able to backtest/validate some more and begin beta testing with a live account!
