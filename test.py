# -*- coding: utf-8 -*-

import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

def plot(data):
    positives = data[data[:, 2] == 1]
    negatives = data[data[:, 2] == 0]

    plt.plot(positives[:, 0], positives[:, 1], 'b+')
    plt.plot(negatives[:, 0], negatives[:, 1], 'yo')

# 绘制SVM决策边界
def visualize_boundary(X, trained_svm):
    kernel = trained_svm.get_params()['kernel']
    if kernel == 'linear':
        w = trained_svm.coef_[0]
        i = trained_svm.intercept_
        xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        a = -w[0] / w[1]
        b = i[0] / w[1]
        yp = a * xp - b
        plt.plot(xp, yp, 'b-')
    elif kernel == 'rbf':
        x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
        
        X1, X2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(np.shape(X1))
        
        for i in range(0, np.shape(X1)[1]):
            this_X = np.c_[X1[:, i], X2[:, i]]
            vals[:, i] = trained_svm.predict(this_X)
        
        plt.contour(X1, X2, vals, colors='blue')

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))

def part1():
    mat = scipy.io.loadmat("dataset_1.mat")
    X, y = mat['X'], mat['y']
    
    plt.title('Data Set 1 Distribution')
    plot(np.c_[X, y])
    plt.show(block=True)
    
    linear_svm = svm.SVC(C=1, kernel='linear')
    linear_svm.fit(X, y.ravel())
    
    plt.title('SVM Decision Boundary with C = 1')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)
    
    linear_svm = svm.SVC(C=100, kernel='linear')
    linear_svm.fit(X, y.ravel())
    
    plt.title('SVM Decision Boundary with C = 100')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)

def part2():
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    print("Similarity between object 1 and object 2: %f" % gaussian_kernel(x1, x2, sigma))
    
    mat_data = scipy.io.loadmat("dataset_2.mat")
    X = mat_data['X']
    y = mat_data['y'].ravel()
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset 2")
    plt.show()
    
    sigma = 0.1
    rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=1.0 / (2 * sigma ** 2))
    rbf_svm.fit(X, y)
    
    plt.title('Nonlinear SVM Decision Boundary')
    plot(np.c_[X, y])
    visualize_boundary(X, rbf_svm)
    plt.show(block=True)

def part3():
    mat = scipy.io.loadmat("dataset_3.mat")
    X, y = mat['X'], mat['y']
    X_val, y_val = mat['Xval'], mat['yval']
    
    plt.title('Dataset 3 Distribution')
    plot(np.c_[X, y])
    plt.show(block=True)
    
    plt.title('Verification Set Distribution')
    plot(np.c_[X_val, y_val])
    plt.show(block=True)
    
    best = params_search(X, y, X_val, y_val)
    rbf_svm = svm.SVC(C=best['C'], kernel='rbf', gamma=best['gamma'])
    rbf_svm.fit(X, y.ravel())
    
    plt.title('Best Parameter SVM Decision Boundary')
    plot(np.c_[X, y])
    visualize_boundary(X, rbf_svm)
    plt.show(block=True)

def params_search(X, y, X_val, y_val):
    c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
    best = {'error': 999, 'C': 0.0, 'sigma': 0.0}
    
    for C in c_values:
        for sigma in sigma_values:
            rbf_svm = svm.SVC(C=C, kernel='rbf', gamma=1.0 / (2 * sigma ** 2))
            rbf_svm.fit(X, y.ravel())
            error = 1 - rbf_svm.score(X_val, y_val)
            
            if error < best['error']:
                best['error'] = error
                best['C'] = C
                best['sigma'] = sigma
    
    best['gamma'] = 1.0 / (2 * best['sigma'] ** 2)
    return best

def main():
    np.set_printoptions(precision=6, linewidth=200)
    part1()
    part2()
    part3()

if __name__ == '__main__':
    main()
