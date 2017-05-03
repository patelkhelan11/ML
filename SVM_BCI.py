#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 14:14:49 2017

@author: khelanpatel
"""

from numpy.testing import assert_array_equal

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy.io
import numpy as np
import csv

import numpy as np
from sklearn import svm

#start subject-1 : Custom SVM
sub1_1=scipy.io.loadmat('train_subject1_psd01.mat')
sub1_X1=sub1_1['X']
sub1_Y1= sub1_1['Y']

sub1_2=scipy.io.loadmat('train_subject1_psd02.mat')
sub1_X2=sub1_2['X']
sub1_Y2= sub1_2['Y']

sub1_3=scipy.io.loadmat('train_subject1_psd03.mat')
sub1_X3=sub1_3['X']
sub1_Y3= sub1_3['Y']

sub1_X=np.concatenate((sub1_X1, sub1_X2, sub1_X3), axis=0)
sub1_Y=np.concatenate((sub1_Y1, sub1_Y2,sub1_Y3), axis=0)

sub1_custom_svm_clf = svm.SVC(C=1.5, cache_size=300, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=4, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
sub1_custom_svm_clf.fit(sub1_X, sub1_Y)

sub1_test=scipy.io.loadmat('test_subject1_psd04.mat')
sub1_X4=sub1_test['X']
sub1_predicted=sub1_custom_svm_clf.predict(sub1_X4)
sub1_Y4=np.loadtxt('test_subject1_true_label.csv',delimiter=",")

print 'sub-1, custom',accuracy_score(sub1_Y4, sub1_predicted)
confusion_matrix(sub1_Y4, sub1_predicted)
#End subject-1 : Custom SVM

#1443/3504 : subject 1 accuracy : simple SVM
#2624/3504 : subject 1 accuracy : Linear SVM
#2192/3504 : subject 1 accuracy : different SVM values
#SVC(C=1.5, cache_size=300, class_weight=None, coef0=0.0,
 #   decision_function_shape=None, degree=4, gamma='auto', kernel='rbf',
 #   max_iter=-1, probability=False, random_state=None, shrinking=True,
 #   tol=0.001, verbose=False)

#start subject-1: simple SVM
sub1_simple_svm_clf = svm.SVC(decision_function_shape='ovo')
sub1_simple_svm_clf.fit(sub1_X, sub1_Y)

sub1_test=scipy.io.loadmat('test_subject1_psd04.mat')
sub1_X4=sub1_test['X']
sub1_predicted=sub1_simple_svm_clf.predict(sub1_X4)
sub1_Y4=np.loadtxt('test_subject1_true_label.csv',delimiter=",")


print 'sub-1, Simple',accuracy_score(sub1_Y4, sub1_predicted)
confusion_matrix(sub1_Y4, sub1_predicted)
#end subject-1 : Simple SVM : Not good results

#start subject-1 : Linear SVM
sub1_linear_svm_clf = svm.LinearSVC()
sub1_linear_svm_clf.fit(sub1_X, sub1_Y)

sub1_test=scipy.io.loadmat('test_subject1_psd04.mat')
sub1_X4=sub1_test['X']
sub1_predicted=sub1_linear_svm_clf.predict(sub1_X4)
sub1_Y4=np.loadtxt('test_subject1_true_label.csv',delimiter=",")

print 'sub-1, Linear',accuracy_score(sub1_Y4, sub1_predicted)
confusion_matrix(sub1_Y4, sub1_predicted)

import pandas as pd
y_actu = pd.Series(sub1_Y4, name='Actual')
y_pred = pd.Series(sub1_predicted, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
import matplotlib.pyplot as plt

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion,title="Confusion matrix for subject-1 with Linear SVM")
#end subject-1 : linear SVM

#start subject-2 : Custom SVM
sub2_1=scipy.io.loadmat('train_subject2_psd01.mat')
sub2_X1=sub2_1['X']
sub2_Y1= sub2_1['Y']

sub2_2=scipy.io.loadmat('train_subject2_psd02.mat')
sub2_X2=sub2_2['X']
sub2_Y2= sub2_2['Y']

sub2_3=scipy.io.loadmat('train_subject2_psd03.mat')
sub2_X3=sub2_3['X']
sub2_Y3= sub2_3['Y']

sub2_X=np.concatenate((sub2_X1, sub2_X2, sub2_X3), axis=0)
sub2_Y=np.concatenate((sub2_Y1, sub2_Y2,sub2_Y3), axis=0)

sub2_custom_svm_clf = svm.SVC(C=1.5, cache_size=300, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=4, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
sub2_custom_svm_clf.fit(sub2_X, sub2_Y)


sub2_test=scipy.io.loadmat('test_subject2_psd04.mat')
sub2_X4=sub2_test['X']
sub2_predicted=sub2_custom_svm_clf.predict(sub2_X4)
sub2_Y4=np.loadtxt('test_subject2_true_label.csv',delimiter=",")

print 'sub-2, custom',accuracy_score(sub2_Y4, sub2_predicted)
confusion_matrix(sub2_Y4, sub2_predicted)
#End of subject-2 : custom SVM : 1456/3472

#start subject-2 : Linear SVM
sub2_linear_svm_clf = svm.LinearSVC()
sub2_linear_svm_clf.fit(sub2_X, sub2_Y)

sub2_predicted=sub2_linear_svm_clf.predict(sub2_X4)

print 'sub-2, Linear',accuracy_score(sub2_Y4, sub2_predicted)
confusion_matrix(sub2_Y4, sub2_predicted)

#End subject-2:Linear SVM : 2174/3472

#start subject-2 : Simple SVM
sub2_simple_svm_clf = svm.SVC()
sub2_simple_svm_clf.fit(sub2_X, sub2_Y)


sub2_predicted=sub2_simple_svm_clf.predict(sub2_X4)


print 'sub-2, simple',accuracy_score(sub2_Y4, sub2_predicted)
confusion_matrix(sub2_Y4, sub2_predicted)
#End subject-2 : Simple SVM : worst results


#subject 3 : start simple SVM
sub3_1=scipy.io.loadmat('train_subject3_psd01.mat')
sub3_X1=sub3_1['X']
sub3_Y1= sub3_1['Y']

sub3_2=scipy.io.loadmat('train_subject3_psd02.mat')
sub3_X2=sub3_2['X']
sub3_Y2= sub3_2['Y']

sub3_3=scipy.io.loadmat('train_subject3_psd03.mat')
sub3_X3=sub3_3['X']
sub3_Y3= sub3_3['Y']

sub3_X=np.concatenate((sub3_X1, sub3_X2, sub3_X3), axis=0)
sub3_Y=np.concatenate((sub3_Y1, sub3_Y2,sub3_Y3), axis=0)

sub3_simple_svm_clf = svm.SVC()
sub3_simple_svm_clf.fit(sub3_X, sub3_Y)

sub3_test=scipy.io.loadmat('test_subject3_psd04.mat')
sub3_X4=sub3_test['X']
sub3_predicted=sub3_simple_svm_clf.predict(sub3_X4)
sub3_Y4=np.loadtxt('test_subject3_true_label.csv',delimiter=",")

#sub3_match=0
#for i in (range(len(sub3_predicted))):
#    if(sub3_predicted[i]==sub3_Y4[i]):
#        sub3_match+=1
#print sub3_match
#print len(sub3_Y4),len(sub3_predicted)
print 'sub-3, simple',accuracy_score(sub3_Y4, sub3_predicted)
confusion_matrix(sub3_Y4, sub3_predicted)
#1347/3488 : end subject 3 :simple SVM

#subject 3: start custom SVM
sub3_custom_svm_clf = svm.SVC(C=3.5, cache_size=400, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
sub3_custom_svm_clf.fit(sub3_X, sub3_Y)

sub3_predicted=sub3_custom_svm_clf.predict(sub3_X4)

print 'sub-3, custom',accuracy_score(sub3_Y4, sub3_predicted)
confusion_matrix(sub3_Y4, sub3_predicted)
#subject 3: end custom SVM : 1347/3488 (same as above)

#subject 3: start Linear SVM
sub3_linear_svm_clf = svm.LinearSVC()
sub3_linear_svm_clf.fit(sub3_X, sub3_Y)

sub3_predicted=sub3_linear_svm_clf.predict(sub3_X4)

print 'sub-3, Linear',accuracy_score(sub3_Y4, sub3_predicted)
confusion_matrix(sub3_Y4, sub3_predicted)

#subject 3: end Linear SVM : 1825/3488
