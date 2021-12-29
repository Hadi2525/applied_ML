@author: Hadi
"""
#######################
########         This is a demo of student assessment
#######################
# Import libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale = 1.2)

# Import the classroom dataset from the csv file
dataset = pd.read_csv('StudentsPerformance.csv')
dataset.head()

# Data visualization considering the effect of test preparation course
# participation for individual student for two scores
sns.lmplot('math score','writing score',data = dataset, 
           hue = 'test preparation course',palette='Set1', 
           fit_reg=False, scatter_kws={"s": 8})

# Locate students who passed the test preparation course
X_none = dataset.loc[dataset['test preparation course']=='none']
# Locate students who did not pass the test preparation course
X_complete = dataset.loc[dataset['test preparation course']=='completed']
# Data comparison and visualization of the effect of test preparation course
# on math scores
fig = plt.figure(figsize=(10,6))
sns.distplot(X_none[['math score']])
sns.distplot(X_complete[['math score']])
fig.legend(labels=['without test preparation course','with test preparation course'])
plt.title('Math score of students out of 100')
plt.show()
# Data comparison and visualization of the effect of test preparation course
# on writing scores
fig = plt.figure(figsize=(10,6))
sns.distplot(X_none[['writing score']])
sns.distplot(X_complete[['writing score']])
fig.legend(labels=['without test preparation course','with test preparation course'])
plt.title('writing score of students out of 100')
plt.show()

# Generate matrix of features of machine learning model
X = np.array(dataset[['gender','parental level of education','math score','writing score']])
# Genreate dependent variable
y = dataset[['test preparation course']]

# Import necessary modules and objects from sci-kit learn package
from sklearn.preprocessing import LabelEncoder
# Use labelencoder for categorical data
labelencoder_y = LabelEncoder()
y_hat=labelencoder_y.fit_transform(y)
"""y_hat= y_hat.reshape(-1,1)"""

# Generate matrix of features of machine learning model
X = dataset[['gender','parental level of education']]
# Genreate dependent variable

le = LabelEncoder()
X_gen = le.fit_transform(X[['gender']])
X_gen = X_gen.reshape(-1,1)

le_par = LabelEncoder()
X_par = le_par.fit_transform(X[['parental level of education']])
X_par = X_par.reshape(-1,1)

X_math = np.array(dataset[['math score']])

X_ord = np.concatenate([X_gen.T, X_par.T, X_math.T]).T


"""y_hat = np.ravel(y_hat)"""
# test and train split approach for machine learning model inputs
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ord, y_hat, test_size = 0.2, random_state = 42)


# Import SVM module from ski-kit learn
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)

# Import evaluation metrics and matrices
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# Define plotting confusion matrix using matplotlib library
# This function is derived from Cognitiveclass ML1010EN course
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat)
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['passed the test','failed the test'],normalize= False,  title='Confusion matrix')
