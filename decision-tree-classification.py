import glob
import os
import sys
import time
import warnings

import numpy as np

from matplotlib import pyplot as plt

from scipy import ndimage as ndi

from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_validation import train_test_split
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


if __name__ == "__main__":
    # shut up scikit-image and numpy
    warnings.simplefilter("ignore")

    images = gen_file_list(sys.argv[1])
    truths = gen_file_list(sys.argv[2])
    output_dir = os.path.abspath(os.path.join(os.getcwd(), sys.argv[3]))

    assert(len(images) == len(truths))

    # read features
    shape = io.imread(images[0], as_grey=True).shape
    nshp  = (shape[0] - 4, shape[1] - 4)
    g_truths = np.zeros((len(images), 1, nshp[0], nshp[1]))
    feats = np.zeros((len(images), 3, nshp[0], nshp[1]))
    for i in xrange(0, len(images)):
        image = io.imread(images[i], as_grey=True)
        truth = io.imread(truths[i], as_grey=True)

        g_truths[i] = truth[2:nshp[0]+2, 2:nshp[1]+2]

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
        feats[i][0] = l_max
        feats[i][1] = l_std
        feats[i][2] = l_lbp

    x_train, x_test, y_train, y_test = train_test_split(feats, g_truths, test_size=0.2, random_state=0)

    # put data into the right format
    def _reshape_im(arr, shape):
        return np.asarray(arr.swapaxes(1, 3).swapaxes(1, 2), dtype="int32").reshape(shape)

    x_train = _reshape_im(x_train, (-1, 3))
    y_train = _reshape_im(y_train, (-1, 1))

    x_test = _reshape_im(x_test, (-1, 3))
    y_test = _reshape_im(y_test, (-1, 1))

    feats = _reshape_im(feats, (-1, 3))

    terry = tree.DecisionTreeClassifier(min_samples_split=5000, max_depth=4)
    terry.fit(x_train, y_train)
    print(terry.score(x_test, y_test))

    predictions = terry.predict(feats)

    # write shadow segmentation out to disk
    npx = nshp[0] * nshp[1]
    for i in range(0, len(predictions), npx):
        im = predictions[i:i+npx].reshape(nshp)

        fn = os.path.join(output_dir, "output-%s.png" % (zero_pad(str(i / npx))))
        io.imsave(fn, im * 255)

    # visualise decision tree
    labels = ["local_maximum", "standard_deviation", "local_binary_pattern"]
    with open("terry.dot", "w") as dotf:
        tree.export_graphviz(terry, dotf, feature_names=labels, filled=False, proportion=True)
