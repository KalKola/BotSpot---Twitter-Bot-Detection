import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("trainingSet.csv")
df = df.fillna(0)

df.head()

X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
        'protected', 'verified', 'notifications', 'contributors_enabled']]
y = df['bot']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)


# parameter sample value set
test_set = {
        'n_estimators': [10, 50, 100, 200, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [int(i) for i in np.linspace(10, 80, num=8)]
}

print("Test Set Values:")
print(test_set['n_estimators'])
print(test_set['max_features'])
print(test_set['max_depth'])

# initializing our RandomForest model
test_clf = RandomForestClassifier(random_state=10)
# initializing grid_search with f1_macro scoring, 10-fold validation, across all processors
grid_search = GridSearchCV(test_clf, param_grid=test_set, cv=10, scoring='f1_macro', n_jobs=-1)

grid_search.fit(X_train, y_train)

# return optimal
optimal = grid_search.best_params_
print("Optimum Values:")
print(optimal)

