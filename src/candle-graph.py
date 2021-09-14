import plotly.graph_objects as plotly
import new

import pandas as pd
from datetime import datetime

stock_data = pd.read_csv(new.data_filename)

fig = plotly.Figure(data = [plotly.Candlestick(x = stock_data["Date"], open = stock_data["Open"], high = stock_data["High"], low = stock_data["Low"], close = stock_data["Close"])])

fig.show()