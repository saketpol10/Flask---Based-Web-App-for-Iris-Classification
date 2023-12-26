# Import Necessary Libraries
import numpy as np
import pandas as pd
# Load the dataset
dataset = pd.read_csv('IRIS.csv')
dataset.head()
dataset.tail()

dataset.describe()
# Data Preprocessing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode Categorical Variables
'''from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)'''

# Split the dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Scale the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Support Vector Model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 42)

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf']}
grid_search = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
#grid_search = GridSearchCV(estimator=classifier1, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Train the Support Vector Model
classifier.fit(X_train, y_train)

# Make Pickle File of our Model
import pickle
pickle.dump(classifier, open("model.pkl", "wb"))

# Evaluate the Support Vector model
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : ", "\n",cm)
print("Accuracy is :",accuracy_score(y_test, y_pred))

