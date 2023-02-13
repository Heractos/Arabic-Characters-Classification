import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

#importing training and testing sets
train_data = pd.read_csv("/Users/hkrukauskas/Documents/GitHub/ML-final-project/csvTrainImages 13440x1024.csv", header=None).values
train_label = pd.read_csv("/Users/hkrukauskas/Documents/GitHub/ML-final-project/csvTrainLabel 13440x1.csv", header=None)

test_data = pd.read_csv("/Users/hkrukauskas/Documents/GitHub/ML-final-project/csvTestImages 3360x1024.csv", header=None).values
test_label = pd.read_csv("/Users/hkrukauskas/Documents/GitHub/ML-final-project/csvTestLabel 3360x1.csv", header=None)

#Reshaping train_label and tes_label to be 2D array
train_label = np.array(train_label).reshape((train_label.shape[0],1))
test_label = np.array(test_label).reshape((test_label.shape[0],1))

#Normalization
train_data = train_data/255.0
test_data = test_data/255.0

#C values for regularization
c_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

linear_acc = []

for C in c_lst:

    clf = svm.SVC(kernel="linear", C=C)

    clf.fit(train_data,train_label)

    test_pred_label = clf.predict(test_data)

    # accuracy
    linear_acc.append(accuracy_score(y_true=test_label, y_pred=test_pred_label))

    # cm
    # print(confusion_matrix(y_true=test_label, y_pred=test_pred_label))

print(linear_acc)
plt.plot(c_lst, linear_acc, "r-")
plt.grid(True)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.show()

# rbf_acc = []

# for C in c_lst:

#     clf = svm.SVC(kernel="rbf", C=C)

#     clf.fit(train_data,train_label)

#     test_pred_label = clf.predict(test_data)

#     # accuracy
#     rbf_acc.append(accuracy_score(y_true=test_label, y_pred=test_pred_label))

#     # cm
#     # print(confusion_matrix(y_true=test_label, y_pred=test_pred_label))

# print(rbf_acc)
# plt.plot(c_lst, rbf_acc, "y-")
# plt.grid(True)
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# plt.show()

# sigmoid_acc = []

# for C in c_lst:

#     clf = svm.SVC(kernel="sigmoid", C=C)

#     clf.fit(train_data,train_label)

#     test_pred_label = clf.predict(test_data)

#     # accuracy
#     sigmoid_acc.append(accuracy_score(y_true=test_label, y_pred=test_pred_label))

#     # cm
#     # print(confusion_matrix(y_true=test_label, y_pred=test_pred_label))

# print(sigmoid_acc)
# plt.plot(c_lst, sigmoid_acc, "b-")
# plt.grid(True)
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# plt.show()