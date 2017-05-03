#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:37:47 2017

@author: khelanpatel
"""
#test for MLP
#Link: http://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
import scipy.io
import numpy as np
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#start subject-1
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

sub1_clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,verbose=True)
sub1_clf_mlp.fit(sub1_X, sub1_Y)

sub1_test=scipy.io.loadmat('test_subject1_psd04.mat')
sub1_X4=sub1_test['X']
sub1_predicted=sub1_clf_mlp.predict(sub1_X4)
sub1_Y4=np.loadtxt('test_subject1_true_label.csv',delimiter=",")

print 'subject-1,',accuracy_score(sub1_predicted, sub1_Y4)
print confusion_matrix(sub1_Y4, sub1_predicted)
#end of subject-1 : NN

#start subject-2:
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

sub2_clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)
sub2_clf_mlp.fit(sub2_X, sub2_Y)

sub2_test=scipy.io.loadmat('test_subject2_psd04.mat')
sub2_X4=sub2_test['X']
sub2_predicted=sub2_clf_mlp.predict(sub2_X4)
sub2_Y4=np.loadtxt('test_subject2_true_label.csv',delimiter=",")

print 'subject-2,',accuracy_score(sub2_predicted, sub2_Y4)
print confusion_matrix(sub2_Y4, sub2_predicted)
#End subject-2:

#start subject-3:
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

sub3_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
sub3_clf.fit(sub3_X, sub3_Y)

sub3_test=scipy.io.loadmat('test_subject3_psd04.mat')
sub3_X4=sub3_test['X']
sub3_predicted=sub3_clf.predict(sub3_X4)
sub3_Y4=np.loadtxt('test_subject3_true_label.csv',delimiter=",")

print 'subject-3',accuracy_score(sub3_predicted, sub3_Y4)
print confusion_matrix(sub3_Y4, sub3_predicted)
#end subject-3:
