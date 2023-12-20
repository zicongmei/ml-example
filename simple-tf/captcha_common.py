from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import urllib.request
import numpy as np

ttf_url = 'https://raw.githubusercontent.com/lepture/captcha/master/src/captcha/data/DroidSansMono.ttf'
ttf_local = "/tmp/DroidSansMono.ttf"
urllib.request.urlretrieve(ttf_url, ttf_local)

image_size = [60, 160, 3]


class generate_data():
    def __init__(self):
        ttf_local = "/tmp/DroidSansMono.ttf"
        urllib.request.urlretrieve(ttf_url, ttf_local)

        self.image = ImageCaptcha(fonts=[ttf_local])

    def generate(self, input):
        data = self.image.generate(input, format='bmp')
        byte_array = data.read()
        int_values = [x for x in byte_array]
        return int_values[54:]

    def display(self, data):
        image_data = np.empty(image_size)
        for j in range(60):
            for k in range(160):
                for i in range(3):
                    image_data[60-1-j][k][i] = data[i + k * image_size[2] +
                                                    j * image_size[1] * image_size[2]] / 256

        plt.imshow(image_data)
        plt.show()
        # print(image_data.shape)
        # print(image_data[0][0][0])
