import requests
from pyquery import PyQuery
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as seb
import random

# 获取数据集
def get_raw_data(url):
    r = requests.get(url)
    data = r.text
    print(data)
    data = data.replace('\t',' ')
    f = open('pla_data.txt','w')
    f.write(data)
    f.close()

# 加载数据集
def get_data():
    pla_data = np.loadtxt('pla_data.txt')
    print(pla_data)
    x = pla_data[:,:-1]
    y = pla_data[:,-1:]
    x0 = np.ones((x.shape[0],1))
    X = np.hstack((x0,x))
    print(X);print(y)
    return X,y


#构建sigmoid函数
def sign(X,w):
    if np.dot(X,w) > 0:
        return 1
    else:
        return -1

# PLA_naive算法 (naive cycle的意思是，每次发生的错误时，循环不重新开始，而是接着进行)
def train_pla(X,y,updates):
    n = X.shape[0]
    m = X.shape[1]
    w = np.zeros(m)
    counts = 0
    for num in range(updates):
        flag = True
        for index,i in enumerate(X):
            if sign(i,w) != y[index]:
                w += i * y[index]
                counts += 1
                flag = False
            else:
                continue
        if flag:
            break
    return w,counts

# 随机PLA算法
def stochastic_pla(X,y,updates):
    counts_mean = []
    for t in range(2000):
        counts = 0
        n = X.shape[0]
        m = X.shape[1]
        w = np.zeros(m)
        random_seed = np.random.permutation(n)
        X = X[random_seed]
        y = y[random_seed]
        for num in range(updates):
            flag = True
            for index,i in enumerate(X):
                if sign(i,w) != y[index]:
                    w += i * y[index]
                    counts += 1
                    flag = False
                else:
                    continue
            if flag:
                break
        counts_mean.append(counts)
    return np.mean(counts_mean)

# 调整学习率的随机PLA算法
def stochastic_learningrate_pla(X,y,updates,n=1):
    counts_mean = []
    for t in range(2000):
        counts = 0
        n = X.shape[0]
        m = X.shape[1]
        w = np.zeros(m)
        random_seed = np.random.permutation(n)
        X = X[random_seed]
        y = y[random_seed]
        for num in range(updates):
            flag = True
            for index,i in enumerate(X):
                if sign(i,w) != y[index]:
                    w += n*(i * y[index])
                    counts += 1
                    flag = False
                else:
                    continue
            if flag:
                break
        counts_mean.append(counts)
    return np.mean(counts_mean)


# 口袋算法
def pocket_pla(X,y,updates=50):
    errors_mean = []
    for t in range(2000):
        counts = 0
        n = X.shape[0]
        m = X.shape[1]
        w = np.zeros(m)
        # error_list = []
        # test_list = []
        random_seed = np.random.permutation(n)
        X = X[random_seed]
        y = y[random_seed]
        for num in range(updates):
            for index, i in enumerate(X):
                if sign(i, w) != y[index]:
                    w_new = w + X[index] * y[index]
                    error_rate0 = error_test(X,y,w)
                    error_rate1 = error_test(X,y,w_new)
                    if error_rate1 < error_rate0:
                        w = w_new
                else:
                    continue
        errors_mean.append(error_test(X,y,w))
    return np.mean(errors_mean)

# pocket算法 求出最优w
def pock(X,y,updates =50):
    counts = 0
    n = X.shape[0]
    m = X.shape[1]
    w = np.zeros(m)
    w_best = np.zeros(m)
    error_best_rate = error_test(X,y,w_best)
    # error_list = []
    # test_list = []
    random_seed = np.random.permutation(n)
    X = X[random_seed]
    y = y[random_seed]
    for num in range(updates):
        for index, i in enumerate(X):
            if sign(i, w) != y[index]:
                w = w + i * y[index]
                error_rate = error_test(X,y,w)
                if error_rate < error_best_rate:
                    error_best_rate = error_rate
                    w_best = w.copy()
                else:
                    continue
    return w_best

# pocket算法
def pocket(X,y,updates = 50):
    for t in range(2000):
        error_rate_mean = []
        n = X.shape[0]
        m = X.shape[1]
        w = np.zeros(m)
        w_best = np.zeros(m)
        error_best_rate = error_test(X,y,w_best)
        for num in range(updates):
            for index, i in enumerate(X):
                if sign(i, w) != y[index]:
                    w = w + i * y[index]
                    error_rate = error_test(X,y,w)
                    if error_rate < error_best_rate:
                        error_best_rate = error_rate
                        w_best = w.copy()
                    else:
                        continue
        error_rate_mean.append(error_test(X,y,w_best))
    return np.mean(error_rate_mean)

# 测试测试集
def pocket_test(X_train,y_train,X,y,runningtimes):
    n = X_train.shape[0]
    m = X_train.shape[1]
    error_list = []
    for times in range(runningtimes):
        random_sort = np.random.permutation(n)
        X_train =  X_train[random_sort]
        y_train = y_train[random_sort]
        X = X[random_sort]
        y = y[random_sort]
        w_best = pock(X_train,y_train)
        error = error_test(X,y,w_best)
        error_list.append(error)
    return np.mean(error_list)

# (19题) pocket算法 求出最优w
def pock_19(X,y,updates =50):
    counts = 0
    n = X.shape[0]
    m = X.shape[1]
    w = np.zeros(m)
    w_best = np.zeros(m)
    error_best_rate = error_test(X,y,w_best)
    # error_list = []
    # test_list = []
    random_seed = np.random.permutation(n)
    X = X[random_seed]
    y = y[random_seed]
    for num in range(updates):
        for index, i in enumerate(X):
            if sign(i, w) != y[index]:
                w = w + i * y[index]
                error_rate = error_test(X,y,w)

    return w






def error_test(X,y,w):
    counts = 0
    for ind,ii in enumerate(X):
        if sign(ii,w) != y[ind]:
            counts += 1
    error_rate = counts/len(X)
    return error_rate


if __name__ == '__main__':
    url = 'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat'
    get_raw_data(url)







