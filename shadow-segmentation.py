import itertools
import os
import sys
import time
import warnings

import numpy as np

import scipy.ndimage as ndi

import matplotlib.pyplot as plt

import skimage.io as io

from sklearn.cluster import MiniBatchKMeans as k_means

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

from skimage.feature import local_binary_pattern

from skimage.filter import gabor_filter
from skimage.filter import gaussian_filter
from skimage.filter import threshold_otsu


def zero_pad(string, num_zeroes=5):
    zeroes = (num_zeroes - len(string))

    if zeroes > 0:
        return (zeroes * "0") + string
    else:
        return string


def _imshow(*images):
    fig = plt.figure()
    for i in range(0, len(images)):
        im = images[i]
        if im.shape[0] > im.shape[1]:
            ax = fig.add_subplot(1, len(images), i + 1)
        else:
            ax = fig.add_subplot(len(images), 1, i + 1)

        cax = ax.imshow(im, cmap=plt.cm.cubehelix)
        fig.colorbar(cax)


def mask_dominant_label(image, labels):
    hist, bins = np.histogram(labels, n_clusters, [0, n_clusters])
    dominant_label = bins[hist.argmax()]

    print("Dominant label: %d" % (dominant_label))
    mask = labels == dominant_label
    image[mask] = 0

    return mask, image


def cluster_metrics(labels_1, labels_2):
    print("\n".join(
        [
            "Normalized Mutual Information: %f" % (normalized_mutual_info_score(labels_1, labels_2)),
            "Adjusted Rand Score: %f" % (adjusted_rand_score(labels_1, labels_2)),
            "Homogeneity: %f" % (homogeneity_score(labels_1, labels_2)),
            "Completeness: %f" % (completeness_score(labels_1, labels_2))
        ]
    ))


def lbp(image, n=3, method="uniform"):
    return local_binary_pattern(image, P=8*n, R=n, method=method)


def segment_texture(image, n_clusters=15, init=None):
    # blur and take local maxima
    blur_image = gaussian_filter(image, sigma=8)
    blur_image = ndi.maximum_filter(blur_image, size=3)

    # get texture features
    feats = lbp(blur_image, n=5, method="uniform")
    feats_r = feats.reshape(-1, 1)

    # cluster the texture features, reusing initialised centres if already calculated
    params = {"n_clusters": n_clusters, "batch_size": 500}
    if init is not None:
        params.update({"init": init})
    km = k_means(**params)
    clus = km.fit(feats_r)

    # copy relevant attributes
    labels = clus.labels_
    clusters = clus.cluster_centers_

    # reshape label arrays
    labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])

    return (image, blur_image, labels, clusters)


def segment_shadow(image, labels, n_clusters=8):
    img = image.ravel()
    shadow_seg = img.copy()
    for i in range(0, n_clusters):
        # set up array of pixel indices matching cluster
        mask = np.nonzero((labels.ravel() == i) == True)[0]
        if len(mask) > 0:
            if img[mask].var() > 0.005:
                thresh = threshold_otsu(img[mask])
                shadow_seg[mask] = shadow_seg[mask] < thresh
            else:
                shadow_seg[mask] = 0

    shadow_seg = shadow_seg.reshape(*image.shape)
    return shadow_seg


if __name__ == "__main__":
    # shut up scikit-image and numpy
    warnings.simplefilter("ignore")

    images = sys.argv[1:-1]
    output_dir = os.path.abspath(os.path.join(os.getcwd(), sys.argv[-1]))

    times = np.zeros((len(images)))

    i = 0
    shape = io.imread(images[i], as_grey=True).shape
    init = None
    for im_path in images:
        image = io.imread(im_path, as_grey=True)

        s = time.time()

        n_clusters = 8
        image, blur_image, labels, init = segment_texture(image, n_clusters=n_clusters, init=init)
        shadow_seg = segment_shadow(blur_image, labels, n_clusters)

        e = time.time()
        times[i] = (e - s)

        i += 1
        fn = os.path.join(output_dir, "output-%s.png" % (zero_pad(str(i))))
        io.imsave(fn, shadow_seg)

    print("Average time to process a %s image: %f seconds" % (str(shape), np.average(times)))
    print("Total time for %d images: %f seconds" % (len(images), np.sum(times)))
