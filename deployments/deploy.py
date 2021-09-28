
import numpy as np
import matplotlib.pyplot as mpp
import tflite_runtime.interpreter as tflite

tflite_model = "STM-StockPrediction-2021-09-28.tflite" # this will be updated depending on whenever this demo will be launched

interpreter = tflite.Interpreter(model_path = tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"])
results = np.squeeze(output_data)
print("output_data (as a tensor) =",output_data)

fig, ax = mpp.subplots(1, figsize=(12,6))

mpp.show()

# # high/low lines
# mpp.plot([x[idx], x[idx]], [val['low'], val['high']], color='black')
# # open marker
# mpp.plot([x[idx], x[idx]-0.1], [val['open'], val['open']], color='black')
# # close marker
# mpp.plot([x[idx], x[idx]+0.1], [val['close'], val['close']], color='black')
