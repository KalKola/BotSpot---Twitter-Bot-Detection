import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm

df = pd.read_csv("t_set.csv")
df = df.drop("notifications", axis=1)
df = df.drop("contributors_enabled", axis=1)
df = df.fillna(0)

X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
        'protected', 'verified']]
y = df['bot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# instantiate and fit the binary logistic model
clf = LogisticRegression(penalty="l2",
                         C=1.0,
                         max_iter=150,
                         class_weight="balanced",
                         random_state=0,
                         warm_start=False,
                         solver='lbfgs',
                         multi_class='ovr',
                         n_jobs=-1,
                         verbose=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

clf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(clf_matrix)



