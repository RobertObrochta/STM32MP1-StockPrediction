
import numpy as np
import tensorflow as tf

tflite_model = "STM-StockPrediction-2021-09-20.tflite" # this will be updated depending on whenever this demo will be launched

interpreter = tf.lite.Interpreter(model_path = tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"])
print("output_data =",output_data)
