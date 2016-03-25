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


def zero_pad(string, num_zeroes=5):
    zeroes = (num_zeroes - len(string))

    if zeroes > 0:
        return (zeroes * "0") + string
    else:
        return string


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
    return (1, local_binary_pattern(image, P=n, R=8 * n, method=method))


def segment_texture(image, n_clusters=15):
    # blur and take local maxima
    image = gaussian_filter(image, sigma=3)
    blur_image = ndi.maximum_filter(image, size=3)

    # get texture features
    num_feats, feats = lbp(blur_image, n=5, method="uniform")
    feats_r = feats.reshape(-1, num_feats)

    # cluster the texture features
    km = k_means(n_clusters=n_clusters)
    clus = km.fit(feats_r)
    labels = clus.labels_

    # reshape label arrays
    labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])

    return (image, blur_image, labels)


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
    import warnings
    warnings.simplefilter("ignore")

    import time

    images = sys.argv[1:-1]
    output_dir = os.path.abspath(os.path.join(os.getcwd(), sys.argv[-1]))

    times = np.zeros((len(images)))

    i = 0
    shape = io.imread(images[i], as_grey=True).shape
    for im_path in images:
        image = io.imread(im_path, as_grey=True)

        s = time.time()

        n_clusters = 8
        image, blur_image, labels = segment_texture(image, n_clusters=n_clusters)

        shadow_seg = segment_shadow(blur_image, labels, n_clusters)

        e = time.time()
        times[i] = (e - s)

        i += 1
        fn = os.path.join(output_dir, "output-%s.png" % (zero_pad(str(i))))
        io.imsave(fn, shadow_seg)

    print("Average time to process a %s image: %f seconds" % (str(shape), np.average(times)))
    print("Total time for %d images: %f seconds" % (len(images), np.sum(times)))


"""
if __name__ == "__main__":
    # set up input
    path = os.path.abspath(os.path.join(os.getcwd(), sys.argv[1]))
    path_2 = os.path.abspath(os.path.join(os.getcwd(), sys.argv[2]))

    # read images
    image = io.imread(path, as_grey=True)
    image_2 = io.imread(path_2, as_grey=True)

    # cluster images
    n_clusters = 8
    image, blur_image, labels = segment_texture(image, n_clusters=n_clusters)
    image_2, blur_image_2, labels_2 = segment_texture(image_2, n_clusters=n_clusters)

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
"""
