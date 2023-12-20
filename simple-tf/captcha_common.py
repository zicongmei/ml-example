from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import urllib.request
import numpy as np
import multiprocessing
import random
import scipy.io

ttf_url = 'https://raw.githubusercontent.com/lepture/captcha/master/src/captcha/data/DroidSansMono.ttf'
ttf_local = "/tmp/DroidSansMono.ttf"
urllib.request.urlretrieve(ttf_url, ttf_local)

image_dimension = [60, 160, 3]
image_size = image_dimension[0] * image_dimension[1] * image_dimension[2]

default_file_path = '/tmp/data1.mat'


class generate_data():
    def __init__(self, digit_size):
        ttf_local = "/tmp/DroidSansMono.ttf"
        urllib.request.urlretrieve(ttf_url, ttf_local)

        self.image = ImageCaptcha(fonts=[ttf_local])
        self.digit_size = digit_size

    def generate_once(self, input):
        data = self.image.generate(input, format='bmp')
        byte_array = data.read()
        int_values = [x for x in byte_array]
        return int_values[54:]

    def generate_random_once(self, c):
        y = random.randint(0, 10**self.digit_size - 1)
        str_y = str(y)
        while len(str_y) < self.digit_size:
            str_y = '0' + str_y
        x = self.generate_once(str_y)
        return x, y

    def generate_all(self, data_size, file_path=None):
        X = np.empty([data_size, image_size])
        y = np.empty([data_size, 1])

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            result = pool.map(
                self.generate_random_once, range(data_size))
        for i in range(data_size):
            X[i], y[i][0] = result[i][0], result[i][1]

        # for i in range(data_size):
        #     print("generating data #", i)
        #     X[i], y[i][0] = self.generate_random_once()

        mdic = {"X": X, "y": y, "label": "captcha"}
        if file_path is None:
            file_path = default_file_path
        scipy.io.savemat(file_path, mdic)
        return X, y

    def load_data(self, file_path=None):
        if file_path is None:
            file_path = default_file_path
        data = scipy.io.loadmat(file_path)
        X = data['X']
        y = data['y']
        return X, y

    def display(self, data, title=None):
        image_data = np.empty(image_dimension)
        for j in range(60):
            for k in range(160):
                for i in range(3):
                    image_data[60-1-j][k][i] = data[i + k * image_dimension[2] +
                                                    j * image_dimension[1] * image_dimension[2]] / 256

        plt.imshow(image_data)
        plt.title(title)
        plt.show()
        # print(image_data.shape)
        # print(image_data[0][0][0])
