import numpy as np

class Path:
    def __init__(self, point, box, feat, id):
        self.color = np.random.randint(30)
        self.points = [point]
        self.last_box = box
        self.last_feat = feat
        self.id = id

    def add_point(self, point):
        self.points.append(point)

