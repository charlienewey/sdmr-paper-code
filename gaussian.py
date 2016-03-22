import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

import skimage.io as io
io.use_plugin("matplotlib")

from sklearn.cluster import MiniBatchKMeans as k_means

from skimage.feature import local_binary_pattern
from skimage.feature import blob_dog

from skimage.filters import gabor_filter
from skimage.filters import gaussian_filter

from skimage.util import view_as_blocks

# feature extraction methods
def lbp(image, n=3, method="uniform"):
    return (1, local_binary_pattern(image, P=n, R=8 * n, method=method))

def gabor(image, freqs=[0.1, 0.4], thetas=[0, 30, 60, 90, 120, 150]):
    params = list(itertools.product(freqs, thetas))
    features = []
    for freq, theta in params:
        g = gabor_filter(image, freq, theta)
        features.append(g[0] + np.real(g[1]))
    features = np.asarray(features).swapaxes(0, 2)
    return (len(params), features)


# take mode over neighbourhood
def neighbourhoodify(image, nbhd=10):
    avgs = np.zeros(map(lambda x: int(x / nbhd), image.shape))
    blocks = view_as_blocks(image, (nbhd, nbhd))
    for row in range(0, len(blocks)):
        for col in range(0, len(blocks[row])):
            mode = np.bincount(blocks[row,col].ravel()).argmax()
            avgs[row, col] = mode
    return avgs


def _imshow(image):
    plt.figure()
    io.imshow(image)


if __name__ == "__main__":
    # set up input
    path = os.path.abspath(os.path.join(os.getcwd(), sys.argv[1]))
    image = io.imread(path, as_grey=True)

    # downscale and smooth
    blur_image = gaussian_filter(image, 15)  # cv2.pyrUp(cv2.pyrDown(image))

    # get texture features
    num_feats, feats = lbp(blur_image, n=6, method="uniform")
    feats_r = feats.reshape(-1, num_feats)

    # set up batch k-means
    n_classes = 10
    mbkm = k_means(n_classes)

    # cluster the local binary patterns with default settings (k = 8)
    clus = mbkm.fit(feats_r)
    labels = clus.labels_

    # reshape label arrays
    labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])

    # make mask of dominant texture
    hist, bins = np.histogram(labels, n_classes, [0, n_classes])
    dominant_texture = bins[hist.argmax()]

    print("Dominant texture class: %d" % (dominant_texture))
    mask = labels == dominant_texture

    # plot histogram of intensity values across dominant texture
    plt.hist(blur_image[np.where(mask)])

    # display images
    _imshow(blur_image)
    _imshow(labels)
    io.show()
