import numpy as np
import pickle as pkl
import os
import cv2

def resize(imgs, size):
    new_shape = np.shape(imgs)[:-2] + size
    rst = np.ones(new_shape)
    for i in range(np.shape(imgs)[0]):
        rst[i, :] = cv2.resize(imgs[i, :], size)

    return rst

def source_load(label, size):
    """
    return source normal data
    """
    f = open(os.path.join("data", "usps.pkl"), "rb")
    a = pkl.load(f)
    X = a["X_train"]
    y = a["y_train"]

    return np.expand_dims(resize(np.reshape(X[y == label, :], [-1, 16, 16]), size), 1)

def target_load(label, num_train, size):
    """
    return:
    target data for training
    rest of target data for testing, with labels
    """
    f = open(os.path.join("data", "usps.pkl"), "rb")
    a = pkl.load(f)
    X = a["X_test"]
    y = a["y_test"]

    X_normal = X[y == label, :]
    X_abnormal = X[y != label, :]

    X_normal = X_normal[np.random.permutation(X_normal.shape[0]), :]
    tgt_X_train = np.reshape(X_normal[:num_train, :], [-1, 16, 16])

    tgt_X_test = np.reshape(np.vstack((X_normal[num_train:, :], X_abnormal)), [-1, 16, 16])

    n1 = np.shape(X_normal[num_train:, :])[0]
    n2 = np.shape(X_abnormal)[0]

    tgt_y_test = np.ones(n1+n2)
    tgt_y_test[:n1] = 0
    # print ("anomaly portion", np.sum(tgt_y_test)/np.size(tgt_y_test))
    return np.expand_dims(resize(tgt_X_train, size), 1), np.expand_dims(resize(tgt_X_test, size), 1), tgt_y_test
