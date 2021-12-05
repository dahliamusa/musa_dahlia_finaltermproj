#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from sklearn.metrics import confusion_matrix

k = 10 #number of splits for 10-fold cross validation
num_metrics = 16 #number of metrics being evaluated

#Load data from csv file
data = pd.read_csv("diabetes.csv", sep=",")
X = data.iloc[:,:-1]
y = data.iloc[: , -1:]

rf_sums = [0 for i in range(0, num_metrics)]
knn_sums = [0 for i in range(0, num_metrics)]
lstm_sums = [0 for i in range(0, num_metrics)]

def calculate_tpr(tn, fp, fn, tp):
    den = tp + fn
    if den == 0:
        den = 0.01
    return tp/den

def calculate_tnr(tn, fp, fn, tp):
    den = tn + fp
    if den == 0:
        den = 0.01
    return tn/den

def calculate_fpr(tn, fp, fn, tp):
    den = tn + fp
    if den == 0:
        den = 0.01
    return fp/den

def calculate_fnr(tn, fp, fn, tp):
    den = tp + fn
    if den == 0:
        den = 0.01
    return fn/den

def calculate_recall(tn, fp, fn, tp):
    den = tp + fn
    if den == 0:
        den = 0.01
    return tp/den

def calculate_precision(tn, fp, fn, tp):
    den = tp + fp
    if den == 0:
        den = 0.01
    return tp/den
    
def calculate_f1(tn, fp, fn, tp):
    return (2 * tp)/(2 * tp + fp + fn)

#accuracy
def calculate_acc(tn, fp, fn, tp):
    den = tp + fp + fn + tn
    if den == 0:
        den = 0.01
    return (tp + tn)/den

def calculate_error_rate(tn, fp, fn, tp):
    den = tp + fp + fn + tn
    if den == 0:
        den = 0.01
    return (fp + fn)/den
    
#balanced accuracy
def calculate_bacc(tn, fp, fn, tp):
    den1 = tp + fn
    if den1 == 0:
        den1 = 0.01
    den2 = tn + fp
    if den2 == 0:
        den2 = 0.01
    return 0.5*((tp/den1) + (tn/den2))

#true skill statistics
def calculate_tss(tn, fp, fn, tp):
    den1 = tp + fn
    if den1 == 0:
        den1 = 0.001
    den2 = fp + tn
    if den2 == 0:
        den2 = 0.001
    return ((tp / den1) - (fp / den2))

#heidke skill score
def calculate_hss(tn, fp, fn, tp):
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if den == 0:
        den = 0.001
    return ((2 * (tp * tn) - (fp * fn)) / den)

#brier score
def calculate_bs(y_test, y_pred):
    sum = 0
    m = len(y_test)
    for n in range(0, m):
        term = (y_test[n] - y_pred[n])**2
        sum += term
    return sum/m
    
#brier skill score
def calculate_bss(y_pred, bs):
    sum_y = 0
    m = len(y_pred)
    for n in range(0, m):
        sum_y += y_pred[n]
    mean = sum_y / m
    sum_den = 0
    for n in range(0, m):
        sum_den = (y_pred[n] - mean)**2
    den = sum_den / m
    return bs/den

def metrics(results):
    tn, fp, fn, tp = results
    tpr = calculate_tpr(tn, fp, fn, tp)
    tnr = calculate_tnr(tn, fp, fn, tp)
    fpr = calculate_fpr(tn, fp, fn, tp)
    fnr = calculate_fnr(tn, fp, fn, tp)
    recall = calculate_recall(tn, fp, fn, tp)
    precision = calculate_precision(tn, fp, fn, tp)
    f1 = calculate_f1(tn, fp, fn, tp)
    acc = calculate_acc(tn, fp, fn, tp)
    error_rate = calculate_error_rate(tn, fp, fn, tp)
    bacc = calculate_bacc(tn, fp, fn, tp)
    tss = calculate_tss(tn, fp, fn, tp)
    hss = calculate_hss(tn, fp, fn, tp)
    
    return [tn, fp, fn, tp, tpr, tnr, fpr, fnr, recall, precision, f1, acc, error_rate, bacc, tss, hss]

def round_numbers(in_list):
    return [round(num, 2) for num in in_list]


# In[ ]:


#RF
def RF_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train,y_train.values.ravel())
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[True, False]).ravel()
    results = metrics(cm)
    return results

#KNN
def KNN_model(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors = 5)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred, labels=[True, False]).ravel()
    results = metrics(cm)
    
    return results

#LSTM
def LSTM_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(LSTM((1), batch_input_shape = (None, None, 1), return_sequences = False))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=15, batch_size=1, verbose=2)
    y_pred = model.predict(X_test)
    
    #define a threshold to convert values to 0 and 1
    threshold = 0.7
    y_test_converted = np.zeros_like(y_test)
    y_test_converted[y_test > threshold] = 1
    y_pred_converted = np.zeros_like(y_pred)
    y_test_converted[y_pred > threshold] = 1

    cm = confusion_matrix(y_test_converted, y_pred_converted, labels=[True, False]).ravel()
    results = metrics(cm)
    return results
    
def main():
    df = pd.DataFrame(columns=['TN', 'FP', 'FN', 'TP', 'TPR', 'TNR', 'FPR', 'FNR', 'Recall', 'Precision', 'F1', 'Accuracy', 'Error Rate', 'BACC', 'TSS', 'HSS'], index=['RF', 'KNN', 'LSTM'])
    kf = KFold(n_splits = k)
    fold_count = 1
    #for each fold, calculate metrics and add values to the sum
    for train_index , test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf_results = RF_model(X_train, X_test, y_train, y_test)
        knn_results = KNN_model(X_train, X_test, y_train, y_test)
        lstm_results = LSTM_model(X_train, X_test, y_train, y_test)

        for i in range(0, num_metrics):
            rf_sums[i] += rf_results[i]
            knn_sums[i] += knn_results[i]
            lstm_sums[i] += lstm_results[i]
        
        df.loc['RF'] = round_numbers(rf_results)
        df.loc['KNN'] = round_numbers(knn_results)
        df.loc['LSTM'] = round_numbers(lstm_results)
        
        print("--------------------------------------------------------------------------------")
        print("Fold ", fold_count)
        print(df)
        print("--------------------------------------------------------------------------------")
        fold_count += 1
        
    for i in range(0, num_metrics):
        rf_sums[i] /= k
        knn_sums[i] /= k
        lstm_sums[i] /= k
        
    df.loc['RF'] = round_numbers(rf_sums)
    df.loc['KNN'] = round_numbers(knn_sums)
    df.loc['LSTM'] = round_numbers(lstm_sums)
    
    print("--------------------------------------------------------------------------------")
    print("Averages")
    print(df)
    print("--------------------------------------------------------------------------------")
        
main()

