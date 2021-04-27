from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import log_reg
from sklearn.decomposition import PCA

x = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(x.data,x.target, test_size = 0.2, random_state = 1)

'''
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
'''
pca = PCA()
x_train = pca.fit_transform(x_train)[:,:2]
x_test = pca.transform(x_test)[:,:2]

print(x_train.shape)
print(x_test.shape)

weights = log_reg.logistic_regression(x_train, y_train, 500, 5e-10, add_intercept = False)
print(weights)

5*10^-5


final_scores = np.dot( x_test,weights)
preds = np.round(log_reg.sigmoid(final_scores))
accuracy = (preds == y_test).sum().astype(float) / len(preds)

#print("this is training accuracy: ", accuracyT)
print("this is testing accuracy: ", accuracy)