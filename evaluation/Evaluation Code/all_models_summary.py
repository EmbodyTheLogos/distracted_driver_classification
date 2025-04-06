import tensorflow as tf
import pickle
import numpy as np



print("original with GAP")
model = tf.keras.models.load_model('original_model_200_epochs')
print(model.summary())


print("original model without GAP")
model = tf.keras.models.load_model('mod_11_200_epochs')
print(model.summary())

print("mod_1")
model = tf.keras.models.load_model('mod_1_200_epochs')
print(model.summary())

print("mod_2")
model = tf.keras.models.load_model('mod_2_200_epochs')
print(model.summary())

print("mod_3")
model = tf.keras.models.load_model('mod_3_200_epochs')
print(model.summary())

print("mod_4")
model = tf.keras.models.load_model('mod_4_200_epochs')
print(model.summary())


print("mod_5")
model = tf.keras.models.load_model('mod_5_200_epochs')
print(model.summary())


print("mod_6")
model = tf.keras.models.load_model('mod_6_200_epochs')
print(model.summary())


print("mod_7")
model = tf.keras.models.load_model('mod_7_200_epochs')
print(model.summary())


print("mod_8")
model = tf.keras.models.load_model('mod_8_200_epochs')
print(model.summary())


print("mod_9")
model = tf.keras.models.load_model('mod_9_200_epochs')
print(model.summary())


print("mod_10")
model = tf.keras.models.load_model('mod_10_200_epochs')
print(model.summary())

print("new_model_1")
model = tf.keras.models.load_model('new_model_1_200_epochs')
print(model.summary())


print("new_model_2")
model = tf.keras.models.load_model('new_model_2_200_epochs')
print(model.summary())


print("new_model_3")
model = tf.keras.models.load_model('new_model_3_200_epochs')
print(model.summary())

print("depthwise_best")
model = tf.keras.models.load_model('depthwise_best_200_epochs')
print(model.summary())
