import os
import cv2
import numpy as np

PATH = '/home/cs/work/siamese/siamese_tf_mnist/fifa/'
VIDEOS = [
    # 'fifa1/',
    # 'fifa2/',
    # 'fifa3/',
    # 'fifa4/',
    # 'fifa5/',
    'fifa2/'
]



class DataSet:

    def __init__(self, x, y):
        self.images = x
        self.labels = y
        self.length = len(y)

    def next_batch(self, n):
        batch_index = np.random.choice(self.length, size=n, replace=False)
        batch_x = [self.images[i] for i in batch_index]
        batch_y = [self.labels[i] for i in batch_index]
        return batch_x, batch_y



class Player:

    def __init__(self):
        self.all_x = []
        self.all_y = []
        for video in VIDEOS:
            self.load_from(PATH + video)
        self.length = len(self.all_x)
        train_index = np.random.choice(self.length, size=int(self.length * 0.8), replace=False)
        test_index = np.setdiff1d(list(range(self.length)), train_index)
        self.train = DataSet(
            [self.all_x[i] for i in train_index],
            [self.all_y[i] for i in train_index]
        )
        self.test = DataSet(
            [self.all_x[i] for i in test_index],
            [self.all_y[i] for i in test_index]
        )


    def load_from(self, path):
        child_dir = os.listdir(path)
        for i in child_dir:
            pics = os.listdir(path + '/' + i)
            for pic in pics:
                image = cv2.imread(path + '/' + i + '/' + pic)
                image = image.reshape(-1) / 255.0
                self.all_x.append(image)
                self.all_y.append(path[-2:] + i)

