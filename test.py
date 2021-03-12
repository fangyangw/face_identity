import tensorflow as tf
from tensorflow.keras import datasets, layers, models

input_shape = (1, 2, 2, 3)
x = tf.random.normal(input_shape)
print(x)
y = tf.keras.layers.Conv2D(
 12, 2, activation='relu', input_shape=input_shape[1:])(x)
print(y)

# With `dilation_rate` as 2.
# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(
#  2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
# print(y.shape)
#
# # With `padding` as "same".
# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(
#  2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
# print(y.shape)
#
# # With extended batch shape [4, 7]:
# input_shape = (4, 7, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(
#  2, 3, activation='relu', input_shape=input_shape[2:])(x)
# print(y.shape)
