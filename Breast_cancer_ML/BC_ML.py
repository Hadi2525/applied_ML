



from sklearn.datasets import load_breast_cancer

e = load_breast_cancer()

features = e['data']
target = e['target']


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.25,random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred_LogReg = classifier.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classifier.fit(X_train,y_train)

y_pred_KNN = classifier.predict(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train,y_train)

y_pred_SVC = classifier.predict(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred_SVCrbf = classifier.predict(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred_Gauss = classifier.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred_tree = classifier.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

y_pred_randomforest = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix

cm_LogisticRegression = confusion_matrix(y_pred_LogReg,y_test)
cm_KNN = confusion_matrix(y_pred_KNN,y_test)
cm_svckernel = confusion_matrix(y_pred_SVC,y_test)
cm_svcrbf = confusion_matrix(y_pred_SVCrbf,y_test)
cm_Gaussian = confusion_matrix(y_pred_Gauss,y_test)
cm_decisiontree = confusion_matrix(y_pred_tree,y_test)
cm_randomforest = confusion_matrix(y_pred_randomforest,y_test)

import matplotlib.pyplot as plt

