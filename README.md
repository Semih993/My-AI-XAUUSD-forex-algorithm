The main file to open and run is 'live_execute.py'

As long as this file is open, the program fetches OANDA XAUUSD 30M OHLCV candle data every half-hour, analyses the chart, and if an
entry is found, it sends an automated telegram message specifying the direction.

The risk:reward is to be based on recent lows and highs by the user.

Enter own telegram chat id and other credenitals in :
        live_function.py    telegram_chat_ID.py


DISCLAIMER: This is file does not give any financial advise, and does not guarantee profits. This algorithm is to be used at your own risk, 
and this algorithm does not claim any responsibility in any financial losses/gains.

