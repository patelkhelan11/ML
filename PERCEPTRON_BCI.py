from numpy.testing import assert_array_equal

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from seqlearn.perceptron import StructuredPerceptron
import scipy.io
import numpy as np
import csv

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

sub1_clf = StructuredPerceptron(decode='viterbi', lr_exponent=0.1, max_iter=10000, random_state=None, trans_features=False, verbose=0)
sub1_clf.fit(sub1_X, sub1_Y,[len(sub1_Y)])

sub1_test=scipy.io.loadmat('test_subject1_psd04.mat')
sub1_X4=sub1_test['X']
sub1_predicted=sub1_clf.predict(sub1_X4)
sub1_Y4=np.loadtxt('test_subject1_true_label.csv',delimiter=",")

print 'subject-1',accuracy_score(sub1_predicted, sub1_Y4)
print confusion_matrix(sub1_Y4, sub1_predicted)
#3017/3504 : subject 1 accuracy

start subject-2
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

sub2_clf = StructuredPerceptron(decode='viterbi', lr_exponent=0.1, max_iter=5000, random_state=None, trans_features=False, verbose=0)
sub2_clf.fit(sub2_X, sub2_Y, [len(sub2_Y)])


sub2_test=scipy.io.loadmat('test_subject2_psd04.mat')
sub2_X4=sub2_test['X']
sub2_predicted=sub2_clf.predict(sub2_X4)
sub2_Y4=np.loadtxt('test_subject2_true_label.csv',delimiter=",")

print 'subject-2',accuracy_score(sub2_predicted, sub2_Y4)
print confusion_matrix(sub2_Y4, sub2_predicted)
#3035/3472 : subject 2 accuracy

#subject 3
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

sub3_clf = StructuredPerceptron(decode='viterbi', lr_exponent=0.1, max_iter=8000, random_state=None, trans_features=False, verbose=0)
sub3_clf.fit(sub3_X, sub3_Y,[len(sub3_Y)])

sub3_test=scipy.io.loadmat('test_subject3_psd04.mat')
sub3_X4=sub3_test['X']
sub3_predicted=sub3_clf.predict(sub3_X4)
sub3_Y4=np.loadtxt('test_subject3_true_label.csv',delimiter=",")

print 'subject-3',accuracy_score(sub3_predicted, sub3_Y4)
print confusion_matrix(sub3_Y4, sub3_predicted)
#2018/3488 : subject 3 accuracy

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

plot_confusion_matrix(df_confusion,title="Confusion matrix for subject-1 with Structured Perceptron")

list_arrays=np.split(sub1_predicted,len(sub1_predicted)/8)
ans_sub1_2HZ=[]
count=0
for i in list_arrays:
    counts = np.bincount(i)
    ans_sub1_2HZ.append(np.argmax(counts))

sub1_match_2HZ=0
sub1_Y4_2HS=np.loadtxt('test_subject1_true_label_2HS.csv',delimiter=",")

print 'subject-1 (2HZ)',accuracy_score(ans_sub1_2HZ, sub1_Y4_2HS)
confusion_matrix(ans_sub1_2HZ, sub1_Y4_2HS)
