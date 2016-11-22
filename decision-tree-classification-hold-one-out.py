import glob
import os
import sys
import time
import warnings

import numpy as np

from matplotlib import pyplot as plt

from scipy import ndimage as ndi

import sklearn.metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn import tree

from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.util.shape import view_as_windows

from skimage import io


def zero_pad(string, num_zeroes=5):
    zeroes = (num_zeroes - len(string))

    if zeroes > 0:
        return (zeroes * "0") + string
    else:
        return string


def gen_file_list(glb):
    return sorted(glob.glob(os.path.abspath(glb)))


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
    plt.show()


def lbp(image, n=3, method="uniform"):
    return local_binary_pattern(image, P=8*n, R=n, method=method)

def read(fname):
    with open(fname, "r") as in_file:
        return [x.rstrip() for x in in_file.readlines()]


if __name__ == "__main__":
    # shut up scikit-image and numpy
    warnings.simplefilter("ignore")

    # read file paths
    x_train_list = read("/vagrant/random-dataset/x_train.txt")
    y_train_list = read("/vagrant/random-dataset/y_train.txt")

    x_test_list = read("/vagrant/random-dataset/x_test.txt")
    y_test_list = read("/vagrant/random-dataset/y_test.txt")

    assert len(x_train_list) == len(y_train_list)
    assert len(x_test_list) == len(y_test_list)


    # read features
    shape = io.imread(x_train_list[0], as_grey=True).shape
    nshp  = (shape[0] - 4, shape[1] - 4)

    x_train = np.zeros((len(x_train_list), 3, nshp[0], nshp[1]))
    y_train = np.zeros((len(y_train_list), 1, nshp[0], nshp[1]))

    x_test = np.zeros((len(x_test_list), 3, nshp[0], nshp[1]))
    y_test = np.zeros((len(y_test_list), 1, nshp[0], nshp[1]))

    for i in xrange(0, len(x_train)):
        image = io.imread(x_train_list[i], as_grey=True)
        truth = io.imread(y_train_list[i], as_grey=True)

        y_train[i] = truth[2:nshp[0]+2, 2:nshp[1]+2]

        # blur and take local maxima
        blur_image = ndi.maximum_filter(gaussian(image, sigma=8), size=3)

        # features
        l_max_arr = blur_image[2:nshp[0]+2, 2:nshp[1]+2].ravel()
        l_max_bins = np.histogram(l_max_arr, bins=3)
        l_max = np.digitize(l_max_arr, l_max_bins[1]).reshape((nshp[0], nshp[1]))

        try:
            l_std_arr = view_as_windows(blur_image, (5, 5)).std(axis=(2, 3)).ravel()
            l_std_bins = np.histogram(l_std_arr, bins=3)
            l_std = np.digitize(l_std_arr, l_std_bins[1]).reshape((nshp[0], nshp[1]))

            l_lbp = lbp(blur_image, n=5, method="uniform")[2:nshp[0]+2, 2:nshp[1]+2]
        except Exception as e:
            print(x_train_list[i], y_train_list[i])

        # get features
        x_train[i][0] = l_max
        x_train[i][1] = l_std
        x_train[i][2] = l_lbp

    for i in xrange(0, len(x_test)):
        image = io.imread(x_test_list[i], as_grey=True)
        truth = io.imread(y_test_list[i], as_grey=True)

        y_test[i] = truth[2:nshp[0]+2, 2:nshp[1]+2]

        # blur and take local maxima
        blur_image = ndi.maximum_filter(gaussian(image, sigma=8), size=3)

        # features
        l_max_arr = blur_image[2:nshp[0]+2, 2:nshp[1]+2].ravel()
        l_max_bins = np.histogram(l_max_arr, bins=3)
        l_max = np.digitize(l_max_arr, l_max_bins[1]).reshape((nshp[0], nshp[1]))

        l_std_arr = view_as_windows(blur_image, (5, 5)).std(axis=(2, 3)).ravel()
        l_std_bins = np.histogram(l_std_arr, bins=3)
        l_std = np.digitize(l_std_arr, l_std_bins[1]).reshape((nshp[0], nshp[1]))

        l_lbp = lbp(blur_image, n=5, method="uniform")[2:nshp[0]+2, 2:nshp[1]+2]

        # get features
        x_test[i][0] = l_max
        x_test[i][1] = l_std
        x_test[i][2] = l_lbp


    # put data into the right format
    def _reshape_im(arr, shape):
        a = np.asarray(arr.swapaxes(1, 3).swapaxes(1, 2), dtype="int32").reshape(shape)
        return a

    #x_train, x_test, y_train, y_test = train_test_split(feats, g_truths, test_size=0.3, random_state=None)


    # do stuff
    #x_train, x_test = feats[tr], feats[te]
    #y_train, y_test = g_truths[tr], g_truths[te]

    x_train = _reshape_im(x_train, (-1, 3))
    y_train = _reshape_im(y_train, (-1, 1))

    x_test = _reshape_im(x_test, (-1, 3))
    y_test = _reshape_im(y_test, (-1, 1))

    terry = tree.DecisionTreeClassifier(min_samples_split=5000, max_depth=4)
    terry.fit(x_train, y_train)

    print(terry.score(x_test, y_test))
    predictions = terry.predict(x_test)

    # jaccard
    ys = y_test.ravel()
    jaccard = sklearn.metrics.jaccard_similarity_score(ys, predictions)

    # rand
    rand = sklearn.metrics.adjusted_rand_score(ys, predictions)

    print("jaccard score: {}".format(jaccard))
    print("rand score: {}".format(rand))

    # write shadow segmentation out to disk
    #npx = nshp[0] * nshp[1]
    #for i in range(0, len(predictions), npx):
    #    im = predictions[i:i+npx].reshape(nshp)
    #    fn = os.path.join(output_dir, "output-%s.png" % (zero_pad(str(i / npx))))
    #    io.imsave(fn, im * 255)

    # visualise decision tree
    # labels = ["local_maximum", "standard_deviation", "local_binary_pattern"]
    # with open("terry.dot", "w") as dotf:
    #    tree.export_graphviz(terry, dotf, feature_names=labels, filled=False, proportion=True)
