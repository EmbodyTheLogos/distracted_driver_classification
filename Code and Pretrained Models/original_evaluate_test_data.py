import tensorflow as tf
import pickle
import numpy as np


#--------------------------------------------------------------------
# load test data
validation_data = pickle.load(open("test_data.pickle","rb"))

validation_images = []
validation_labels = []
for features, label in validation_data:
  validation_images.append(features)
  validation_labels.append(label)

# Convert to numpy array of type float32 in order to feed into the neural net
validation_images = np.array(validation_images).astype(np.float32)
validation_labels = np.array(validation_labels).astype(np.float32)


model = tf.keras.models.load_model('original_model_200_epochs')
print(model.summary())
print("Original Model 200 epochs - Test Data")
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)





