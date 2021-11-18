
import time
import numpy as np
import matplotlib.pyplot as mpp
import tflite_runtime.interpreter as tflite
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk

tflite_model = "STM-StockPrediction-2021-09-28.tflite" # this will be updated depending on whenever this demo will be launched

interpreter = tflite.Interpreter(model_path = tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"]) # still need to interpret this in the context of an actual price
results = np.squeeze(output_data)
print("Output_data (as a tensor) =",output_data)

# Gtk object to make the final result appear on screen (for now, it is just the adj. closing price)
class OutputText(Gtk.Window):
    def __init__(self):
        super().__init__(title = "StockPrediction")
        hbox = Gtk.Box(spacing=20)
        hbox.set_homogeneous(True)
        gtk_provider = Gtk.CssProvider()
        css = """#output {     
                            background-color: #F5DFA7;
                            font-size: 70px; 
                            color: #628B97;
                          }""" # styling the text


        label = Gtk.Label(label = str(output_data))
        label.set_name("output")

        gtk_provider.load_from_data(css.encode())

        hbox.add(label)
        self.add(hbox)

        gtk_context = Gtk.StyleContext()
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), gtk_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        


window = OutputText()
window.connect("destroy", Gtk.main_quit)
window.show_all()
window.fullscreen()
# ADD: close application window after 30 seconds or on double touch
Gtk.main()












'''
Commented out, matplotlib doesn't load up on the target device. Gonna find another way to show a result (using Gtk for now)
'''
# fig, ax = mpp.subplots(1, figsize=(12,6))

# mpp.show()

# # high/low lines
# mpp.plot([x[idx], x[idx]], [val['low'], val['high']], color='black')
# # open marker
# mpp.plot([x[idx], x[idx]-0.1], [val['open'], val['open']], color='black')
# # close marker
# mpp.plot([x[idx], x[idx]+0.1], [val['close'], val['close']], color='black')
