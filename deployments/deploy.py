
import numpy as np
import matplotlib.pyplot as mpp
import tflite_runtime.interpreter as tflite
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib


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


        label = Gtk.Label(label = str(results))
        label.set_name("output")

        gtk_provider.load_from_data(css.encode())

        hbox.add(label)
        self.add(hbox)

        gtk_context = Gtk.StyleContext()
        gtk_context.add_provider_for_screen(Gdk.Screen.get_default(), gtk_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)


def interpret_tflite():
  tflite_model = "STM-StockPrediction-2021-09-28.tflite" 

  interpreter = tflite.Interpreter(model_path = tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_shape = input_details[0]["shape"]
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]["index"], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]["index"])
  results = np.squeeze(output_data) # adjust for interpretation of price
  print("Results (as a tensor) =", results)

  return results


def main():
  invoke_interpreter = interpret_tflite()

  window = OutputText(invoke_interpreter)
  window.connect("destroy", Gtk.main_quit)
  window.show_all()
  GLib.timeout_add(30000, Gtk.main_quit, window) # exits the Gtk main loop after 30 seconds
  window.fullscreen() 
  Gtk.main()


main()
