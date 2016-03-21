import itertools
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans as k_means
from skimage.feature import local_binary_pattern
from skimage.feature import blob_dog
from skimage.filters import gabor_filter
from skimage.filters import gaussian_filter

# feature extraction methods
def lbp(image, n=3, method="uniform"):
    return local_binary_pattern(image, n, 8 * n, method=method)

def gabor(image, freqs=[0.4], theta=[0, 30, 60, 90, 120, 150]):
    f = gabor_filter(image, 0.1, 120)
    return f[0] + np.sqrt(f[1] ** 2)  # real and imaginary parts

def neighbourhoodify(image):
    # neighbourhood-ify
    nbhd = 5
    avgs = np.zeros((labels.shape[0] / nbhd, labels.shape[1] / nbhd))
    i, j = (0, 0)
    for x in range(0, labels.shape[0] - nbhd, nbhd):
        for y in range(0, labels.shape[1] - nbhd):
            if j >= labels.shape[1] / nbhd:
                j = 0
            patch = labels[x:x+nbhd,y:y+nbhd][0]
            mode = np.bincount(patch).argmax()
            avgs[i,j] = mode
            j += 1
        i += 1
    return avgs


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

    # downscale and smooth
    blur_image = cv2.pyrUp(cv2.pyrDown(image))

    # get texture features
    feat = lbp(blur_image, n=15)
    print("Feature dimensions: %d x %d" % feat.shape)
    feat_r = feat.reshape((feat.shape[0] * feat.shape[1], 1))
    print("Flattening feature array: %d x %d" % feat.shape)

    # set up batch k-means
    n_classes = 15
    mbkm = k_means(n_classes)

    # cluster the local binary patterns with default settings (k = 8)
    clus = mbkm.fit(feat_r)
    labels = clus.labels_

    print("Labels shape: %d" % labels.shape)

    # reshape label arrays
    labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])
    print("Reshaping label array: %d x %d" % labels.shape)

    # make mask of dominant texture
    hist, bins = np.histogram(labels.ravel(), 256, [0, n_classes])
    dominant_texture = bins[hist.argmax()]
    mask = labels == dominant_texture

    # plot intensity values across dominant texture
    plt.hist(image[np.where(mask)])

    # change intensity values so that the texture can be easily seen
    image[mask] = 255

    # make texture classes more visible and display them
    imshow(image,
           labels * int(255 / n_classes))
