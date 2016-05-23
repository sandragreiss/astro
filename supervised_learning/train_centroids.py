import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation

BRISK_THRESHOLD = 30
BRISK_OCTAVES = 3
BRISK_SCALE = 1.0
N_CLUSTERS = 128

PATH = '/data/sandragreiss/astro/data/'


def _load_image(image_path):
    return cv2.imread(image_path, 0)


def _features(cv_image):
    """ Takes a cvMat grayscale image and return the corresponding
        brisk descriptors. Do not change the brisk parameters """

    brisk = cv2.BRISK(BRISK_THRESHOLD,
                      BRISK_OCTAVES,
                      BRISK_SCALE)

    try:
        _, descriptors = brisk.detectAndCompute(cv_image, None)
    except:
        return []

    return descriptors


def calculate_centroids(descriptors):
    """ Calculated a brisk histogram (128-vector) for `image_url.` """
    km = KMeans(n_clusters=N_CLUSTERS)
    km.fit(descriptors)
    centroids = km.cluster_centers_

    return centroids, km


def get_descriptors_sample(descriptors, sample_size):
    descriptors = np.vstack(descriptors)
    np.random.shuffle(descriptors)
    return descriptors[:sample_size]


def _get_all_descriptors(directory, tag):
    files = os.listdir(PATH + directory)
    print len(files)
    total = 0
    all_descriptors = []
    tags = []

    for fle in files:
        total += 1
        image_path = PATH + directory + fle
        image_arr = _load_image(image_path)
        descriptors = _features(image_arr)
        all_descriptors.append(descriptors)
        tags.append(tag)
        if total % 1000 == 0:
            print '1000 done'
            print len(tags)

    return all_descriptors, tags


def calculate_histograms(km, all_descriptors):
    hists_by_image = []
    for descriptors in all_descriptors:
        cids = km.predict(descriptors)
        hist = np.bincount(cids, minlength=128)
        hists_by_image.append(hist)

    hists_by_image = np.vstack(hists_by_image)

    return hists_by_image


da_descriptors, da_tags = _get_all_descriptors('DA_images/', 'DA')
print len(da_tags)
nonda_descriptors, nonda_tags = _get_all_descriptors('non_DA_images/', 'non_DA')
print len(nonda_tags)
all_descriptors = np.concatenate((da_descriptors, nonda_descriptors))
all_tags = np.concatenate((da_tags, nonda_tags))

descriptors_subset = get_descriptors_sample(all_descriptors, 10000)

centroids, km = calculate_centroids(descriptors_subset)


with open('centroids_astro.py', 'w') as output:
    for c in centroids:
        output.write('%s \n' % c)

print 'done centroids'


histograms = calculate_histograms(km, all_descriptors)

print histograms.shape
"""
with open('histograms.py', 'w') as output:
    for h in histograms:
        output.write('%s \n' % h)

print 'done hists'

with open('tags.py', 'w') as output:
    for t in all_tags:
        output.write('%s \n' % t)

print 'done tags'

datafile = open('histograms_tmp.py', 'r') 
tokens = datafile.read().split(',\n')
all_histograms = []
for token in tokens:
    l = token.replace('\n', '')
    l_tmp = np.asarray(ast.literal_eval(l))
    all_histograms.append(l_tmp)

all_histograms = np.vstack(all_histograms)

tags_file = open('tags.py', 'r')
all_tags = np.asarray(tags_file.read().split(' \n'))
"""
#clf_linearsvc = LinearSVC(loss='hinge', penalty='l2', random_state=42, max_iter=10000, verbose=1)
#scores_linear = cross_validation.cross_val_score(clf_linearsvc, histograms, all_tags, cv=5)

#print("Accuracy linear: %0.2f (+/- %0.2f)" % (scores_linear.mean(), scores_linear.std() * 2))

clf_rbfsvc = SVC(C=100, gamma=0.01, kernel='rbf', max_iter=10000, shrinking=True, tol=0.001,
                 verbose=True)

scores_nonlinear = cross_validation.cross_val_score(clf_rbfsvc, histograms, all_tags, cv=5)
print("Accuracy non-linear: %0.2f (+/- %0.2f)" % (scores_nonlinear.mean(), scores_nonlinear.std() * 2))

from sklearn.grid_search import GridSearchCV

C_range = np.logspace(-2, 4, 5)
gamma_range = np.logspace(-3, 3, 5)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = cross_validation.StratifiedShuffleSplit(all_tags, n_iter=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(kernel='rbf', verbose=True), param_grid=param_grid, cv=5)
grid.fit(histograms, all_tags)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf_rbfsvc = SVC(C=10, gamma=0.001, kernel='rbf', max_iter=10000, shrinking=True, tol=0.001,
                 verbose=True)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(histograms,
                                                                     all_tags,
                                                                     test_size=0.2,
                                                                     random_state=0)

clf_rbfsvc.fit(X_train, y_train)
y_pred = clf_rbfsvc.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

sum_indexes = []

for hist in histograms:
    index_tmp = []
    for index, item in enumerate(hist):
        if item > 10:
            index_tmp.append(index)
    sum_indexes.append(np.sum(index_tmp))
