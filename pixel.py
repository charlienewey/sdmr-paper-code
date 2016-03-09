import itertools
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel

# feature extraction methods
def lbp(image, n=3, method="uniform"):
    return local_binary_pattern(image, n, 8 * n, method=method)

def gabor(image, freqs=[0.4], theta=[0, 30, 60, 90, 120, 150], sigmas=[3]):
    i = image
    for f, t, s in itertools.product(freqs, theta, sigmas):
        kern = np.real(gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s))
        i = ndi.convolve(k, kern, mode="wrap")
    return i


def imshow(*images):
    # create the figure
    fig = plt.figure(figsize=(10, 10))
    for i in range(0, len(images)):
        # display original image with locations of patchea
        ax = fig.add_subplot(len(images), 1, i + 1)
        ax.imshow(images[i], cmap=plt.cm.gray, interpolation="nearest", vmin=0, vmax=255)
    # display the patches and plot
    plt.show()


if __name__ == "__main__":
    # set up input
    path = os.path.abspath(os.path.join(os.getcwd(), sys.argv[1]))
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print("Image dimensions: %d x %d" % image.shape)

    # get texture measures
    feat = lbp(image)
    print("Feature dimensions: %d x %d" % feat.shape)

    feat_r = feat.reshape((feat.shape[0] * feat.shape[1], 1))
    print("Flattening feature array: %d x %d" % feat_r.shape)

    # set up batch k-means
    n_classes = 10
    mbkm = MiniBatchKMeans(n_classes)

    # cluster the local binary patterns with default settings (k = 8)
    clus = mbkm.fit(feat_r)
    labs = clus.labels_
    print("Labels shape: %d" % labs.shape)

    # reshape label array
    labs = labs.reshape(image.shape[0], image.shape[1])
    print("Reshaping label array: %d x %d" % labs.shape)

    # neighbourhood-ify
    #nbhd = 5
    #avgs = np.zeros((labs.shape[0] / nbhd, labs.shape[1] / nbhd))
    #print(avgs.shape)
    #i, j = (0, 0)
    #for x in range(0, labs.shape[0] - nbhd, nbhd):
    #    for y in range(0, labs.shape[1] - nbhd):
    #        if j >= labs.shape[1] / nbhd:
    #            j = 0
    #        patch = labs[x:x+nbhd,y:y+nbhd][0]
    #        mode = np.bincount(patch).argmax()
    #        avgs[i,j] = mode
    #        j += 1
    #    i += 1

    # make texture classes more visible and display them
    labs *= int(255 / n_classes)  # exaggerate values so that the difference is obvious
    imshow(image, labs)
