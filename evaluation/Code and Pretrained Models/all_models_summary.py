import tensorflow as tf
import pickle
import numpy as np



print("\noriginal with GAP")
model = tf.keras.models.load_model('original_model_200_epochs')
print(model.summary())


print("\noriginal model without GAP")
model = tf.keras.models.load_model('original_model_without_GAP__200_epochs')
print(model.summary())

print("\nmod_1")
model = tf.keras.models.load_model('mod_1_200_epochs')
print(model.summary())

print("\nmod_2")
model = tf.keras.models.load_model('mod_2_200_epochs')
print(model.summary())

print("\nmod_3")
model = tf.keras.models.load_model('mod_3_200_epochs')
print(model.summary())

print("\nmod_4")
model = tf.keras.models.load_model('mod_4_200_epochs')
print(model.summary())


print("\nmod_5")
model = tf.keras.models.load_model('mod_5_200_epochs')
print(model.summary())


print("\nmod_6")
model = tf.keras.models.load_model('mod_6_200_epochs')
print(model.summary())


print("\nmod_7")
model = tf.keras.models.load_model('mod_7_200_epochs')
print(model.summary())


print("\nmod_8")
model = tf.keras.models.load_model('mod_8_200_epochs')
print(model.summary())


print("\nmod_9")
model = tf.keras.models.load_model('mod_9_200_epochs')
print(model.summary())


print("\nmod_10")
model = tf.keras.models.load_model('mod_10_200_epochs')
print(model.summary())

print("\nnew_model_1")
model = tf.keras.models.load_model('new_model_1_200_epochs')
print(model.summary())


print("\nnew_model_2")
model = tf.keras.models.load_model('new_model_2_200_epochs')
print(model.summary())


print("\nnew_model_3")
model = tf.keras.models.load_model('new_model_3_200_epochs')
print(model.summary())

print("\ndepthwise_best")
model = tf.keras.models.load_model('depthwise_best_200_epochs')
print(model.summary())
