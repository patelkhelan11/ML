# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io
import numpy as np
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mat = scipy.io.loadmat('train_subject1_psd01.mat')
#print mat['X']
#print mat['Y']

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

clf = LinearDiscriminantAnalysis()
clf.fit(new_X, new_Y)
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
ans=clf.predict(new_X)
test_ans=clf.predict(test_X)


print 'subject-1',accuracy_score(true_label, test_ans)
confusion_matrix(true_label, test_ans)
#8382/10432 accuracy in the subject_1 training set.
#2378/3504 accuracy in the subject_1 for final. 67%

#start of code for subject-2
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

sub2_clf = LinearDiscriminantAnalysis()
sub2_clf.fit(sub2_X, sub2_Y)
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)


sub2_test=scipy.io.loadmat('test_subject2_psd04.mat')
sub2_X4=sub2_test['X']
sub2_predicted=sub2_clf.predict(sub2_X4)
sub2_Y4=np.loadtxt('test_subject2_true_label.csv',delimiter=",")

print 'subject-2',accuracy_score(sub2_predicted, sub2_Y4)
confusion_matrix(sub2_predicted, sub2_Y4)

# 2019/3472

#start of code for subject-3
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

sub3_clf = LinearDiscriminantAnalysis()
sub3_clf.fit(sub3_X, sub3_Y)
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)


sub3_test=scipy.io.loadmat('test_subject3_psd04.mat')
sub3_X4=sub3_test['X']
sub3_predicted=sub3_clf.predict(sub3_X4)
sub3_Y4=np.loadtxt('test_subject3_true_label.csv',delimiter=",")

print 'subject-3',accuracy_score(sub3_predicted, sub3_Y4)
confusion_matrix(sub3_predicted, sub3_Y4)
#1713/3488

#test code for 2HZ-subject3
list_arrays=np.split(sub3_predicted,len(sub3_predicted)/8)
ans_sub3_2HZ=[]
count=0
for i in list_arrays:
    counts = np.bincount(i)
    ans_sub3_2HZ.append(np.argmax(counts))

sub3_match_2HZ=0
sub3_Y4_2HS=np.loadtxt('test_subject3_true_label_2HS.csv',delimiter=",")

print 'subject-1 (2HZ)',accuracy_score(ans_sub3_2HZ, sub3_Y4_2HS)
confusion_matrix(ans_sub3_2HZ, sub3_Y4_2HS)
#219/436 for 2HZ-subject 3

#test code for 2HZ-subject1
list_arrays_1=np.split(test_ans,len(test_ans)/8)
ans_sub1_2HZ=[]
count_1=0
for i in list_arrays_1:
    counts_1 = np.bincount(i)
    ans_sub1_2HZ.append(np.argmax(counts_1))

sub1_match_2HZ=0
sub1_Y4_2HS=np.loadtxt('test_subject1_true_label_2HS.csv',delimiter=",")

print 'subject-2 (2HZ)',accuracy_score(ans_sub1_2HZ, sub1_Y4_2HS)
confusion_matrix(ans_sub1_2HZ, sub1_Y4_2HS)
#308/438 for 2HZ-subject 1

#test code for 2HZ-subject2
list_arrays=np.split(sub2_predicted,len(sub2_predicted)/8)
ans_sub2_2HZ=[]
count=0
for i in list_arrays:
    counts = np.bincount(i)
    ans_sub2_2HZ.append(np.argmax(counts))

sub2_match_2HZ=0
sub2_Y4_2HS=np.loadtxt('test_subject2_true_label_2HS.csv',delimiter=",")

print 'subject-3 (2HZ)',accuracy_score(ans_sub2_2HZ, sub2_Y4_2HS)
confusion_matrix(ans_sub2_2HZ, sub2_Y4_2HS)
#253/434 for 2HZ-subject2
