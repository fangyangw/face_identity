from tensorflow import keras
from matplotlib import pyplot as plt
from numpy.linalg import svd
import numpy as np
a = np.random.randint(-10, 10, (4, 3)).astype(float)
u, s, vh = np.linalg.svd(a)
print(u, s, vh)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
