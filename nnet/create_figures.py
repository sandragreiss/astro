import os
import matplotlib.pylab as plt
import numpy as np

PATH = '/Users/sgreiss/sandra/machine_learning/warwick_seminar/data/'

LAMBDA_MIN = 3820
LAMBDA_MAX = 9000
NORM_MIN = 0
NORM_MAX = 1


def _create_figure(x, y, directory, fle):
    plt.plot(x, y, 'k')
    plt.axis([3500, 9500, 0, 1.25])
    plt.tight_layout()
    plt.savefig(directory + fle + '.png')
    plt.close()

    print fle + ' done'


def _get_data(directory):
    files = os.listdir(PATH + directory)
    print len(files)
    total = 0

    for fle in files:
        x = np.loadtxt(PATH + directory + fle, usecols=[0])
        y = np.loadtxt(PATH + directory + fle, usecols=[1])

        if x.min() < LAMBDA_MIN and x.max() > LAMBDA_MAX:
            total += 1
            x = x[(x >= LAMBDA_MIN) & (x <= LAMBDA_MAX)]
            y = y[(x >= LAMBDA_MIN) & (x <= LAMBDA_MAX)]

            A = y.min()
            B = y.max()

            normalized_y = (NORM_MIN + (y - A) * (NORM_MAX - NORM_MIN)) / (B - A)

            saved_file = fle.replace('.dat', '')
            saved_directory = PATH + 'non_DA_images/'

            _create_figure(x, normalized_y, saved_directory, saved_file)

    return total

print _get_data('non-DA/')
