import itertools
import os
import sys
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

from skimage.exposure import equalize_hist

from skimage.feature import local_binary_pattern

from skimage.filters import gabor_filter
from skimage.filters import gaussian_filter
from skimage.filters import threshold_otsu

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


def _imshow(*images):
    fig = plt.figure()
    for i in range(0, len(images)):
        im = images[i]["image"]
        if im.shape[0] > im.shape[1]:
            ax = fig.add_subplot(1, len(images), i + 1)
        else:
            ax = fig.add_subplot(len(images), 1, i + 1)

        if "cmap" in images[i]:
            cmap = images[i]["cmap"]
        else: cmap = plt.cm.inferno

        cax = ax.imshow(images[i]["image"], cmap=cmap)
        fig.colorbar(cax)


def segment(path, n_clusters=15):
    image = io.imread(path, as_grey=True)

    # blur and take local maxima
    image = gaussian_filter(image, sigma=8)
    blur_image = ndi.maximum_filter(image, size=3)

    # get texture features
    num_feats, feats = lbp(blur_image, n=5, method="uniform")
    feats_r = feats.reshape(-1, num_feats)

    # set up batch k-means
    km = k_means(n_clusters=n_clusters)

    # cluster the local binary patterns with default settings (k = 8)
    clus = km.fit(feats_r)
    labels = clus.labels_

    # reshape label arrays
    labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])

    return (image, blur_image, labels)


def mask_dominant_label(image, labels):
    # make mask of dominant texture
    hist, bins = np.histogram(labels, n_clusters, [0, n_clusters])
    dominant_texture = bins[hist.argmax()]

    print("Dominant texture label: %d" % (dominant_texture))
    mask = labels == dominant_texture
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


def segment_shadow(image, labels):
    img = np.copy(image.ravel())
    shadow_seg = np.copy(img)
    for i in range(0, n_clusters):
        # set up array of pixel indices matching cluster
        mask = np.nonzero((labels.ravel() == i) == True)[0];
        # if there are pixels with this label...
        if len(mask) > 0:
            if img[mask].var() > 0.005:
                thresh = threshold_otsu(img[mask])
                shadow = shadow_seg < thresh
                shadow_seg[mask] = shadow[mask]
            else:
                shadow_seg[mask] = 0

    shadow_seg = shadow_seg.reshape(*image.shape)
    return shadow_seg


if __name__ == "__main__":
    # set up input
    path = os.path.abspath(os.path.join(os.getcwd(), sys.argv[1]))
    path_2 = os.path.abspath(os.path.join(os.getcwd(), sys.argv[2]))

    # cluster images
    n_clusters = 8
    image, blur_image, labels = segment(path, n_clusters=n_clusters)
    image_2, blur_image_2, labels_2 = segment(path_2, n_clusters=n_clusters)

    # mask dominant textures
    mask, blur_image = mask_dominant_label(blur_image, labels)
    mask_2, blur_image_2 = mask_dominant_label(blur_image_2, labels_2)

    # segment shadows
    shadow_seg = segment_shadow(image, labels)
    shadow_seg_2 = segment_shadow(image_2, labels_2)

    # plot histograms of intensity values across dominant texture
    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    ax.hist(image[np.where(mask)])

    ax = fig.add_subplot(2, 1, 2)
    ax.hist(image_2[np.where(mask_2)])

    # get and print cluster metrics
    print("Dominant cluster metrics: ")
    cluster_metrics(mask.ravel(), mask_2.ravel())

    print("\nOverall cluster metrics: ")
    cluster_metrics(labels.ravel(), labels_2.ravel())

    # show figures
    masked = image_2.copy()
    masked[shadow_seg_2 == 1] = 0
    _imshow(
        {"image": shadow_seg_2},
        {"image": image_2},
        {"image": masked}
    )

    _imshow(
        {"image": shadow_seg},
        {"image": image}
    )

    io.show()
