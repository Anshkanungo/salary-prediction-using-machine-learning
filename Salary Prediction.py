# Improting Important files:-

from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('adult.csv')

# First we need a header column for our data:-

headerList = [
    'Age', 'Workclass', 'Fnlwgt', 'Education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]
df.to_csv("adult2.csv", header=headerList, index=False)
file = pd.read_csv('adult2.csv', na_values='?')
file = file.assign(capita=file['capital_gain'] - file['capital_loss'])
file.head()

print(file)

# Shape of the data:-

print("\n")
print("Shape of the data is:")
print(file.shape)
print(" ")

# dealing with Null values and Duplicate values:-

print(" ")
print("null values are: ")
print(file.isnull().sum())
file.drop_duplicates(inplace=True)

# Changing all the 'object' class to 'int' class. And exchanging last two columns as we need our results in the last column:-

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
file['income'] = lb.fit_transform(file['income'])
file['Workclass'] = lb.fit_transform(file['Workclass'])
file['native_country'] = lb.fit_transform(file['native_country'])
file['sex'] = lb.fit_transform(file['sex'])
file['occupation'] = lb.fit_transform(file['occupation'])
file['marital_status'] = lb.fit_transform(file['marital_status'])
file['Education'] = lb.fit_transform(file['Education'])
file['race'] = lb.fit_transform(file['race'])

colList = list(file.columns)
colList[15], colList[14] = colList[14], colList[15]
file = file[colList]

# dropping unnecessary columns:-

file.drop('Fnlwgt', axis=1, inplace=True)
file.drop('education_num', axis=1, inplace=True)
file.drop('relationship', axis=1, inplace=True)

print(file)

#Classifying the data:-

x = file.iloc[:, :-1]
y = file.iloc[:, -1]

print(" ")
print(type(x))
print(type(y))
print(" ")

#Splitting the data in test data and training data:-

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
print(f"Training data X: {x_train.shape}")
print(f"Testing data X: {x_test.shape}")
print(f"Training data Y: {y_train.shape}")
print(f"Testing data Y: {y_test.shape}")

# Fucntion to Give Results:-

from sklearn.metrics import confusion_matrix, classification_report

# A function to tell model's accuracy and Classification report:-


def apply_model(model, x_train, x_test, y_train, y_test, Name):
    model.fit(x_train, y_train)
    ypred = model.predict(x_test)
    print(" ")
    print("\n\n------------------------------------------------------------")
    print(f"{Name} Model Results:")
    print("\n------------------------------------------------------------")
    print(" ")
    print(f'Training Score {model.score(x_train, y_train)}')
    print(f'Testing Score {model.score(x_test, y_test)}')
    cm = confusion_matrix(y_test, ypred)
    print(f'Confusion_matrix {cm}\n')
    print(f'Classification_report {classification_report(y_test, ypred)}\n', )
    return (model.score(x_test, y_test))


""""
We will be applying all these models:
    a. Decision Tree Classifier
    b. Random Forest Classifier
    c. KNN Classifier
    d. Logistic Regression
    e. SVM Classifier
    But we will use Hyper-parameter tuning (RandomizedSearchCv) to get the best parameters for each model to get the   best Individual results.
"""

from sklearn.model_selection import RandomizedSearchCV

test_values = []

# A: Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
params_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 12, 14, 16, 18, 25, 30],
    'min_samples_split': [10, 12, 14, 16, 18, 20, 22, 24]
}

rscv1 = RandomizedSearchCV(dt, param_distributions=params_dt)
r1 = apply_model(rscv1, x_train, x_test, y_train, y_test,
                 "DecisionTreeClassifier")
test_values.append(r1)

# B: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
params_rf = {
    'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 13, 15, 17, 18],
    'min_samples_split': [12, 14, 16, 28, 20, 22, 24]
}
rscv2 = RandomizedSearchCV(rf, param_distributions=params_rf)
r2 = apply_model(rscv2, x_train, x_test, y_train, y_test,
                 "RandomForestClassifier")
test_values.append(r2)

# C: KNNClassifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
params_knn = {'n_neighbors': list(range(10, 80, 5))}
rscv3 = RandomizedSearchCV(knn, param_distributions=params_knn)
r3 = apply_model(rscv3, x_train, x_test, y_train, y_test,
                 "KNeighborsClassifier")
test_values.append(r3)

# D: Logistic Regression

from sklearn.linear_model import LogisticRegression

rscv4 = LogisticRegression(solver='liblinear')
r4 = apply_model(rscv4, x_train, x_test, y_train, y_test, "LogisticRegression")
test_values.append(r4)
"""
E: SVM

from sklearn.svm import SVC

svm = SVC(kernel="linear", C=100)
r5 = apply_model(svm, x_train, x_test, y_train, y_test, "SVM Classifier")
test_values.append(r5)

# As this took way to much time(97 min when i stopped), i decided to remove this model.

"""

# Checking the best model with best accuracy:-

test_keys = [
    "Decision Tree Classifier", " Random Forest Classifier", "KNN Classifier",
    "Logistic Regression", "SVM Classifier"
]

res = dict(zip(test_keys, test_values))

Keymax = max(res, key=lambda x: res[x])
print("\n\n------------------------------------------------------------")
print(f"\nbest accuracy found is {max(test_values)} of model {Keymax}")
print("\n------------------------------------------------------------")