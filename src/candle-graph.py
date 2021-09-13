import plotly.graph_objects as plotly
import main

import pandas as pd
from datetime import datetime

stock_data = pd.read_csv(main.data_filename)

fig = plotly.Figure(data = [plotly.Candlestick(x = stock_data['date'], open = stock_data['open'], high = stock_data['high'], low = stock_data['low'], close = stock_data['close'])])

fig.show()