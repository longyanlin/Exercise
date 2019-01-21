import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')

examples_n = 100  # display some images
indexes = np.random.choice(range(mnist.train.images.shape[0]), examples_n, replace=False)

fig = plt.figure(figsize=(5, 5))

for i in range(1, examples_n + 1):
    a = fig.add_subplot(np.sqrt(examples_n), np.sqrt(examples_n), i)
    a.axis('off')
    image = mnist.train.images[indexes[i - 1]].reshape((28, 28))
    a.imshow(image, cmap='Greys_r')

plt.show()