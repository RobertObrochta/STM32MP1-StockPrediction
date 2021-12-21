
import numpy as np
import tflite_runtime.interpreter as tflite
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

from datetime import date
from dateutil.relativedelta import *
import csv


today_date = date.today()
data_filename = f"STM-{today_date}.csv"
tflite_filename = f"STM-StockPrediction-{today_date}.tflite"

# Gtk object to make the final result appear on screen (for now, it is just the adj. closing price)
class OutputText(Gtk.Window):
    def __init__(self, results):
        super().__init__(title = "StockPrediction")
        hbox = Gtk.Box(spacing=20)
        hbox.set_homogeneous(True)
        gtk_provider = Gtk.CssProvider()
        css = """#output {     
                            background-color: #F5DFA7;
                            font-size: 70px; 
                            color: #628B97;
                          }""" # styling the text


        label = Gtk.Label(label = "$" + str(results))
        label.set_name("output")

        gtk_provider.load_from_data(css.encode())

        hbox.add(label)
        self.add(hbox)

        gtk_context = Gtk.StyleContext()
        gtk_context.add_provider_for_screen(Gdk.Screen.get_default(), gtk_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


def parse_csv():
  test_file = data_filename

  # error catching. defaults to 2021-12-20 file if not found
  try:
    reader = csv.reader(open(test_file, "r"), delimiter=',')
  except FileNotFoundError:
    test_file = "STM-2021-12-20.csv"
    reader = csv.reader(open(test_file, "r"), delimiter=',')

  next(reader)
  x = []
  for item in reader:
    no_dates = item[1:]
    x.append(no_dates)
  train_set = np.array(x, dtype = np.float32)[0:50]

  features = train_set.copy()
  features = np.array(features, dtype = np.float32)[0:50]

  return train_set


def interpret_tflite():
  tflite_model = tflite_filename
  features = parse_csv()

  # error catching. defaults to 2021-12-20.tflte file
  try:
    interpreter = tflite.Interpreter(model_path = tflite_model)
  except ValueError:
    tflite_model = "STM-StockPrediction-2021-12-20.tflite"
    interpreter = tflite.Interpreter(model_path = tflite_model)

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  interpreter.resize_tensor_input(input_details[0]["index"], [1, 50, 6])
  interpreter.allocate_tensors()

  input_data10 = np.expand_dims(features, axis = 0)
  
  interpreter.set_tensor(input_details[0]["index"], input_data10)
  interpreter.invoke()

  print(output_details)
  output_data = interpreter.get_tensor(output_details[0]["index"])
  results = np.squeeze(output_data)
  print("Results (as a tensor) =", results)

  calculated_result = results * 100 - 10
  print(f"Interpreted result (as adj. closing price): ${calculated_result}")
  
  return calculated_result


def main():
  invoke_interpreter = interpret_tflite()

  window = OutputText(invoke_interpreter)
  window.connect("destroy", Gtk.main_quit)
  window.show_all()
  GLib.timeout_add(30000, Gtk.main_quit, window) # exits the Gtk main loop after 30 seconds
  window.fullscreen() 
  Gtk.main()


main()
