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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def svm_pred():

    # read in training set
    df = pd.read_csv("datasets/t_set.csv")
    # replace Dataframe NaN values with 0 for nominal features
    df = df.fillna(0)

    X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
            'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
            'protected', 'verified']]
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
    clf = svm.SVC(kernel='linear', probability=True, cache_size=2000, random_state=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # save model accuracy metrics
    svm_acc_raw = metrics.accuracy_score(y_test, y_pred)
    svm_acc_mat = confusion_matrix(y_test, y_pred)
    svm_class_rep = classification_report(y_test, y_pred)

    print("-- Model Complete --")

    # Generate ROC-AUC Curve - Primarly for System Evaluation
    roc_prod = 1
    print("Produce ROC-AUC Curve? (System Evaluation)")
    print(" 1. Yes")
    print(" 2. No")

    roc_prod = input()
    if roc_prod == '1':

        # calculate AUC, FPR, TRP, & Threshhold levels
        svm_prob = clf.predict_proba(X_test)
        svm_prob = svm_prob[:, 1]
        svm_auc = roc_auc_score(y_test, svm_prob)
        print("AUC Value: " + str(svm_auc))
        fpr, tpr, thresh = roc_curve(y_test, svm_prob)

        # create AUC-ROC Curve for SVM model
        plt.style.use('ggplot')
        plt.plot([0,1], [0,1], linestyle='--', color='darkblue', label='Baseline')
        plt.plot(fpr, tpr, color="darkorange", label='ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC-ROC SVM Curve')
        plt.legend()
        plt.show()
    else:
        print("returning to menu")

    return clf, svm_acc_raw, svm_acc_mat, svm_class_rep
