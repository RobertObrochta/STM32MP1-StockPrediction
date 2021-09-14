import plotly.graph_objects as plotly
import new
import pandas as pd
from time import sleep

stock_data = pd.read_csv(new.data_filename)
# had to load the predicted data into nested lists, as plotly wants the data in a list
load_predicted = [[new.date_with_outlook], [new.future_open], [new.future_high], [new.future_low], [new.future_close]]
 
all_data = plotly.Figure(data = [plotly.Candlestick(x = stock_data["Date"], open = stock_data["Open"], high = stock_data["High"], low = stock_data["Low"], close = stock_data["Close"])])
predicted_data = plotly.Figure(data = [plotly.Candlestick(x = load_predicted[0], open = load_predicted[1], high = load_predicted[2], low = load_predicted[3], close = load_predicted[4])])

all_data.show()
sleep(60)
predicted_data.show()
