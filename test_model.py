import tensorflow as tf
import matplotlib.pyplot as plt


def getmax(predictions):
    values = []
    for index, prediction in enumerate(predictions):
        max = 0
        value = -1
        for i, v in enumerate(prediction):
            if v > max:
                max = v
                value = i
        values.append(value)
    return values

model_save_path = "test.model"
model = tf.keras.models.load_model(model_save_path)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

for test_images, test_labels in test_ds:
    predictions = model(test_images)
    values1 = getmax(predictions)
    values = tf.argmax(predictions, 1)
    for index, value in enumerate(values):
        vv = int(test_labels[index])
        a = 1
        if vv != value:
            print("%s prediction: %s" % (str(vv), value))
            plt.imshow(test_images[index])
            plt.title("%s prediction: %s" % (str(vv), value))
    plt.show()
    # test_step(test_images, test_labels)

print(test_loss.result(), test_accuracy.result()*100)

if __name__ == '__main__':
    pass