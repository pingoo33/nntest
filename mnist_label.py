import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import mnist

from model.mnist_cnn import MnistCNN

if __name__ == "__main__":
    dir_list = ['acgan', 'began', 'bgan', 'cgan', 'dcgan', 'ebgan', 'lsgan', 'relativistic_gan', 'sgan', 'wgan_gp']

    model = MnistCNN('mnist_cnn')
    model.load_model()

    for dir in dir_list:
        path = './images/' + dir + '/gen'
        file_names = os.listdir(path)

        for file in file_names:
            img_path = os.path.join(path, file)
            img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
            image_tensor = image.img_to_array(img)
            image_tensor = image_tensor.reshape((28, 28, 1))
            # image_tensor = image_tensor.reshape((-1, 28, 28, 1))
            # image_tensor = image_tensor[0]
            image_tensor /= 255

            output = model.get_prob(image_tensor)
            output = np.argmax(output)

            name = file.split('_')[0]
            name = name + '_' + str(output) + '.png'

            dst_path = os.path.join(path, name)
            os.rename(img_path, dst_path)
