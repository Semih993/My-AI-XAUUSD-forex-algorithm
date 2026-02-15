The main file to open and run is 'live_execute.py'

As long as this file is open, the program fetches OANDA XAUUSD 30M OHLCV candle data every half-hour, analyses the chart, and if an
entry is found, it sends an automated telegram message specifying the direction.

The risk:reward is to be based on recent lows and highs by the user.

