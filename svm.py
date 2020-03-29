import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler


def svm_pred():

    df = pd.read_csv("t_set.csv")
    df = df.fillna(0)

    X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
            'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
            'protected', 'verified', 'notifications', 'contributors_enabled']]
    y = df['bot']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

    # using MinMaxScaler() to limit the boundary space
    # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    # X_train = scaling.transform(X_train)
    # X_test = scaling.transform(X_test)

    # using StandardScaler() to limit boundary space
    # scaled = StandardScaler()
    # X_train = scaled.fit_transform(X_train)
    # X_test = scaled.fit_transform(X_test)

    # using RobustScaler() to limit boundary space
    scaled = RobustScaler()
    X_train = scaled.fit_transform(X_train)
    X_test = scaled.fit_transform(X_test)

    # using preprocessing to normalize the data
    # X_train = preprocessing.scale(X_train)
    # X_test = preprocessing.scale(X_test)

    # generate SVM model
    print("Building Support Vector Machine Model...")
    clf = svm.SVC(kernel='linear', cache_size=2000, random_state=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    svm_acc_raw = metrics.accuracy_score(y_test, y_pred)
    svm_acc_mat = confusion_matrix(y_test, y_pred)
    svm_class_rep = classification_report(y_test, y_pred)

    print("-- Model Complete --")

    return clf, svm_acc_raw, svm_acc_mat, svm_class_rep
