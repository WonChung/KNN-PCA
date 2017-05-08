# MNIST handwritten digit recognition - data file loading demo
# Written by Matt Zucker, April 2017

# You can tell this script was written by Won because he imports sys
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import numpy as np
import gzip
import struct
from sklearn.decomposition import PCA
import datetime
import time

IMAGE_SIZE = 28

######################################################################
# Read a 32-bit int from a file or a stream

def read_int(f):
    buf = f.read(4)
    data = struct.unpack('>i', buf)
    return data[0]

######################################################################
# Open a regular file or a gzipped file to decompress on-the-fly

def open_maybe_gz(filename, mode='rb'):

    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)


# OpenCV has fast matching code, but the Python interface to it
# changes significantly from version to version. This is a reasonably
# fast pure numpy k-nearest-neighbor function that you might find
# helpful for your own code.

#points = search space
#p = center, point to search around
#k = number of neighbors to return
def bruteforce_knn(points, p, k):

    assert(len(p) == points.shape[1])

    diff = points - p
    d = (diff**2).sum(axis=1)
    idx = np.argpartition(d, k)

    idx = idx[:k]
    d = d[idx]

    idx2 = np.argsort(d)
    return idx[idx2], np.sqrt(d[idx2])

######################################################################
# Read the MNIST data from an images file or a labels file. The file
# formats are documented at http://yann.lecun.com/exdb/mnist/

def read_mnist(images_file, labels_file):

    images = open_maybe_gz(images_file)

    imagic = read_int(images)
    assert(imagic == 2051)
    icount = read_int(images)
    rows = read_int(images)
    cols = read_int(images)
    assert(rows == IMAGE_SIZE and cols == IMAGE_SIZE)

    print 'reading', icount, 'images of', rows, 'rows by', cols, 'cols.'

    labels = open_maybe_gz(labels_file)

    lmagic = read_int(labels)
    assert(lmagic == 2049)
    lcount = read_int(labels)

    print 'reading', lcount, 'labels.'

    assert(icount == lcount)

    image_array = np.fromstring(images.read(icount*rows*cols),
                                dtype=np.uint8).reshape((icount,rows,cols))

    label_array = np.fromstring(labels.read(lcount),
                                dtype=np.uint8).reshape((icount))

    return image_array, label_array

######################################################################
# Show use of the MNIST data set:

def main():


    # Read images and labels. This is reading the 10k-element test set
    # (you can also use the other pair of filenames to get the
    # 60k-element training set).
    unknownImages, unknownLabels = read_mnist('MNIST_data/t10k-images-idx3-ubyte.gz',
                                'MNIST_data/t10k-labels-idx1-ubyte.gz')
    knownImages, knownLabels = read_mnist('MNIST_data/train-images-idx3-ubyte.gz',
                                          'MNIST_data/train-labels-idx1-ubyte.gz')

    # This is a nice way to reshape and rescale the MNIST data
    # (e.g. to feed to PCA, Neural Net, etc.) It converts the data to
    # 32-bit floating point, and then recenters it to be in the [-1,
    # 1] range.
    classifier_input_unknown = unknownImages.reshape(-1, IMAGE_SIZE * IMAGE_SIZE).astype(np.float32)
    classifier_input_unknown = classifier_input_unknown * (2.0 / 255.0) - 1.0

    classifier_input_known = knownImages.reshape(-1, IMAGE_SIZE * IMAGE_SIZE).astype(np.float32)
    classifier_input_known = classifier_input_known * (2.0 / 255.0) - 1.0

    # Test Start Time
    test_start = datetime.datetime.now()
    test_startSeconds = time.mktime(test_start.timetuple())

    # Number of Components to Extract
    pca = PCA(n_components=30)

    # Fit the Training Data
    pca.fit(classifier_input_known)

    #Extract Feature
    ExtractedKnown= pca.fit_transform(classifier_input_known)
    ExtractedUnknown = pca.transform(classifier_input_unknown)

    k = 3 #set k for knn
    numCorrect = 0 #used to calculate accuracy
    for i, image in enumerate(unknownImages):
        #old def of p
        #p = unknownImages[i,:,:].flatten() #change this so p = flatten(image) #change to test data
        p = ExtractedUnknown[i,:] #i think its row-col and not col-row
        #pdb.set_trace()

        matches, dist = bruteforce_knn(ExtractedKnown, p, k)
        #pdb.set_trace()

        #create a voting histogram, which holds the # of votes for each class
        voting = np.zeros(10)
        for j in range(k):
            voting[knownLabels[matches[j]]] = voting[knownLabels[matches[j]]] + 1

        #find the classification with the most votes (if tie, take the first appearance)
        max = 0
        classification = 0
        for j in range(10):
            if max < voting[j]:
                max = voting[j]
                classification = j

        #need to find the max occuring
        # print('the point {} was classified as {}'.format(labels[i], labelsTraining[matches[0]]))
        print('the point {} was classified as {}'.format(unknownLabels[i], classification))
        #todo: fix classification accuracy printout
        #if unknownLabels[i]==knownLabels[matches[0]]:
        if unknownLabels[i]==classification:
            numCorrect = numCorrect+1
        print('classification accuracy: {}'.format(float(numCorrect)/(i+1)))


    # Test End Time
    test_end = datetime.datetime.now()
    test_endSeconds = time.mktime(test_end.timetuple())
    print "Testing: ", test_endSeconds-test_startSeconds
######################################################################

if __name__ == '__main__':
    main()
