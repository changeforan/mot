import numpy as np
from . import tools
from siamese import siamese_network


class Sampler:
    def __init__(self, feat_conv, image_np):
        self.feature_map = np.squeeze(feat_conv)
        self.image_np = image_np


    def sample(self, box):
        pass



class SiameseSampler(Sampler):

    def __init__(self):
        Sampler.__init__(self, None, None)
        self.model = siamese_network.Siamese()


    def sample(self, box, image_np):
        player_img = tools.get_player_img(box, image_np)
        feat = self.model.run(player_img)
        return np.squeeze(feat)


class FeatSampler(Sampler):

    def sample(self, box):
        original_feat = FeatSampler.get_feat_on_feature_map(box, self.feature_map)
        return FeatSampler.sampling(original_feat)

    @staticmethod
    def get_feat_on_feature_map(box, feature_map):
        ymin, xmin, ymax, xmax = box
        len_y, len_x, _ = feature_map.shape
        return feature_map[
               int(ymin * len_y):int(ymax * len_y) + 1,
               int(xmin * len_x):int(xmax * len_x) + 1
               ]

    @staticmethod
    def sampling(original_feat):
        len_y, len_x, deep = original_feat.shape
        feat = []
        cut_y = 8
        cut_x = 4
        for i in range(deep):
            for y in range(cut_y):
                for x in range(cut_x):
                    ymin = int(y * len_y * 1.0 / cut_y)
                    ymax = int((y + 1) * len_y * 1.0 / cut_y)
                    xmin = int(x * len_x * 1.0 / cut_x)
                    xmax = int((x + 1) * len_x * 1.0 / cut_x)
                    feats = original_feat[ymin:ymax + 1, xmin:xmax + 1, i]
                    feat.append(np.max(feats))
        return feat