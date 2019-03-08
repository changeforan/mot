from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from inputdata import  Player

def visualize(embed, x_test):

    # two ways of visualization: scale to fit [0,1] scale
    # feat = embed - np.min(embed, 0)
    # feat /= np.max(feat, 0)

    # two ways of visualization: leave with original scale
    feat = embed
    ax_min = np.min(embed,0)
    ax_max = np.max(embed,0)
    ax_dist_sq = np.sum((ax_max-ax_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        dist = np.sum((feat[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-5*ax_dist_sq:   # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [feat[i]]]
        patch_to_color = x_test[i]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(patch_to_color, zoom=0.5, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False
        )
        ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    # plt.xticks([]), plt.yticks([])
    plt.title('Embedding from the last layer of the network')
    plt.show()

if __name__ == "__main__":

    players = Player()
    x_test = players.test.images
    x_test = np.array(x_test).reshape([-1, 28, 28, 3]) * 255
    x_test = x_test.astype(int)
    embed = np.fromfile('embed.txt', dtype=np.float32)
    embed = embed.reshape([-1, 2])
    visualize(embed, x_test)