# https://celikmustafa89.medium.com/converting-tensorflow-pb-model-to-tflite-model-a-step-by-step-guide-de8bb9c2d24e

import tensorflow as tf
# Load the .pb file
converter = tf.lite.TFLiteConverter.from_saved_model("original_model_200_epochs")
# Convert the model
tflite_model = converter.convert()

# Save the .tflite file
with open("original_model_200_epochs.tflite", "wb") as f:
   f.write(tflite_model)

# Load the .pb file
converter = tf.lite.TFLiteConverter.from_saved_model("depthwise_best_200_epochs")
# Convert the model
tflite_model = converter.convert()

# Save the .tflite file
with open("depthwise_best_200_epochs.tflite", "wb") as f:
   f.write(tflite_model)