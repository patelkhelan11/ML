#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 19:46:39 2017

@author: khelanpatel
"""

import scipy.io
import numpy as np
import csv
from seqlearn.hmm import MultinomialHMM
from seqlearn.evaluation import whole_sequence_accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from numpy.testing import assert_array_equal
import scipy.io
import numpy as np
import csv

mat = scipy.io.loadmat('train_subject1_psd01.mat')
print mat['X']
print mat['Y']

mat1=scipy.io.loadmat('train_subject1_psd02.mat')
X1=mat1['X']
Y1=mat1['Y']

mat2=scipy.io.loadmat('train_subject1_psd03.mat')
X2=mat1['X']
Y2=mat1['Y']

mat_test=scipy.io.loadmat('test_subject1_psd04.mat')
test_X=mat_test['X']
true_label=np.loadtxt('test_subject1_true_label.csv',delimiter=",")

X=mat['X']
Y=mat['Y']

new_X=np.concatenate((X, X1, X2), axis=0)
new_Y=np.concatenate((Y, Y1, Y2), axis=0)

clf = MultinomialHMM()
clf.fit(new_X, new_Y, len(new_X))
clf.set_params(decode="bestfirst")
ans=clf.predict(test_X)

print 'sub-1, custom',accuracy_score(ans, true_label)
print confusion_matrix(true_label, ans)
#1440/3504: subject 1 accuracy
#start subject-2
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

clf_2 = MultinomialHMM()
clf_2.fit(sub2_X, sub2_Y, len(sub2_X))
clf_2.set_params(decode="bestfirst",alpha=0)

sub2_test=scipy.io.loadmat('test_subject2_psd04.mat')
sub2_X4=sub2_test['X']
sub2_Y4=np.loadtxt('test_subject2_true_label.csv',delimiter=",")

ans_2=clf_2.predict(sub2_X4)

print 'sub-1, custom',accuracy_score(ans_2, sub2_Y4)
print confusion_matrix(sub2_Y4, ans_2)
#1456/3472 : subject 2 accuracy

#start subject-3
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

clf_3 = MultinomialHMM()
clf_3.fit(sub3_X, sub3_Y, len(sub3_X))
clf_3.set_params(decode='viterbi', alpha=1.5)

sub3_test=scipy.io.loadmat('test_subject3_psd04.mat')
sub3_X4=sub3_test['X']
sub3_Y4=np.loadtxt('test_subject3_true_label.csv',delimiter=",")

ans_3=clf_3.predict(sub3_X4)

print 'sub-3, custom',accuracy_score(ans_3, sub3_Y4)
print confusion_matrix(sub3_Y4, ans_3)
#1120/3488 : subject-3 accuracy
#1200/3488 : subject-3 accuracy with best-fit

#print whole_sequence_accuracy(sub3_Y4,ans_3,len(ans_3))
