import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def lr_pred():

    # read in training set
    df = pd.read_csv("datasets/t_set.csv")
    df = df.drop("notifications", axis=1)
    df = df.drop("contributors_enabled", axis=1)
    # replace Dataframe NaN values with 0 for nominal features
    df = df.fillna(0)

    X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
            'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
            'protected', 'verified']]
    y = df['bot']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("Building Logistic Regression Model...")
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

    lr_acc_raw = metrics.accuracy_score(y_test, y_pred)
    lr_acc_mat = confusion_matrix(y_test, y_pred)
    lr_class_rep = classification_report(y_test, y_pred)

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
        plt.plot([0, 1], [0, 1], linestyle='--', color='darkblue', label='Baseline')
        plt.plot(fpr, tpr, color="darkorange", label='ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC-ROC LR Curve')
        plt.legend()
        plt.show()

    return clf, lr_acc_raw, lr_acc_mat, lr_class_rep
