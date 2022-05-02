import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

train_data = pd.read_csv("/Users/hkrukauskas/Downloads/archive/csvTrainImages 13440x1024.csv", header=None).values
train_label = pd.read_csv("/Users/hkrukauskas/Downloads/archive/csvTrainLabel 13440x1.csv", header=None)

test_data = pd.read_csv("/Users/hkrukauskas/Downloads/archive/csvTestImages 3360x1024.csv", header=None).values
test_label = pd.read_csv("/Users/hkrukauskas/Downloads/archive/csvTestLabel 3360x1.csv", header=None)

train_label = np.array(train_label).reshape((train_label.shape[0],1))
test_label = np.array(test_label).reshape((test_label.shape[0],1))

train_data = train_data/255.0
test_data = test_data/255.0

clf = svm.SVC()

clf.fit(train_data,train_label)

test_pred_label = clf.predict(test_data)

# accuracy
print("accuracy:", accuracy_score(y_true=test_label, y_pred=test_pred_label), "\n")

# cm
print(confusion_matrix(y_true=test_label, y_pred=test_pred_label))