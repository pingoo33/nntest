import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from model.mnist_cnn import MnistCNN


def calculate_variation(data):
    sum_y = 0

    for y in data:
        sum_y += y

    mean = sum_y / len(data)

    square_sum = 0
    for y in data:
        square_sum += (y - mean) ** 2

    variation = square_sum / len(data)
    return mean, variation


if __name__ == "__main__":
    dir_list = ['acgan', 'began', 'cgan', 'dagan', 'dcgan', 'ebgan', 'lsgan', 'relativistic_gan', 'sgan',
                'wgan_gp']

    model = MnistCNN('mnist_cnn')
    model.load_model()

    for dir_name in dir_list:
        path = './images/' + dir_name + '/gen'
        file_names = os.listdir(path)

        defects = [0] * 10
        killed_class = [False] * 10
        for file in file_names:
            img_path = os.path.join(path, file)
            img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
            img_tensor = image.img_to_array(img)
            img_tensor = img_tensor.reshape((28, 28, 1))
            img_tensor /= 255

            label = file.split('.')[0]
            name = label.split('_')[0]
            label = int(label.split('_')[1])

            output = model.get_prob(img_tensor)
            output = np.argmax(output)

            if label >= 10:
                print(dir_name)
                print(name)

            if label != output:
                killed_class[label] = True
                defects[label] += 1

        defects = np.array(defects)
        df = pd.DataFrame(defects)

        f = open('output/mnist_cnn/%s_stat.txt' % dir_name, 'w')
        for i, defect in enumerate(defects):
            f.write('%d: %d   ' % (i, defect))

        num_killed_class = 0
        for c in killed_class:
            if c:
                num_killed_class += 1

        score = num_killed_class / 10.0

        mean, variation = calculate_variation(defects)
        f.write('\nmutation score: %f' % score)
        f.write('\nmean: %f' % mean)
        f.write('\nvariation: %f' % variation)
        f.close()

        title = 'defects of ' + dir_name
        ax = df.plot(kind='bar', figsize=(10, 6))
        ax.set_xlabel('label')
        ax.set_ylabel('number of defects')
        plt.savefig('output/mnist_cnn/%s_defects.png' % dir_name)
        plt.clf()
