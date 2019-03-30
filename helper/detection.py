class Detection:
    def __init__(self, location, feat_cnn, feat_sim, width, box):
        self.location = location
        self.feat_cnn = feat_cnn
        self.feat_sim = feat_sim
        self.width = width
        self.box = box

    def change_location(self, new_location, flag=False):
        if flag:
            dy = new_location[0] - self.location[0]
            dx = new_location[1] - self.location[1]
            self.box[0] += dy
            self.box[1] += dx
            self.box[2] += dy
            self.box[3] += dx
        self.location = new_location

