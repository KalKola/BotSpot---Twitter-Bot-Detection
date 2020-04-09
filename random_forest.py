import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def rf_pred():

    df = pd.read_csv("t_set.csv")
    df = df.fillna(0)

    X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
            'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
            'protected', 'verified', 'notifications', 'contributors_enabled']]
    y = df['bot']

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    print("Building Random Forest Model...")
    # initializing the RF model with optimized parameters - rfTest()
    clf = RandomForestClassifier(n_estimators=1000,
                                 max_depth=20,
                                 max_features='auto',
                                 oob_score='TRUE',
                                 bootstrap='TRUE',
                                 random_state=10)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # generating accuracy metrics, confusion matrix, and classification report
    rf_acc_raw = metrics.accuracy_score(y_test, y_pred)
    rf_acc_mat = confusion_matrix(y_test, y_pred)
    rf_class_rep = classification_report(y_test, y_pred)

    print("-- Model Complete --")

    roc_prod = 1
    print("Produce ROC-AUC Curve? (Testing Purpose)")
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
        plt.title('AUC-ROC RF Curve')
        plt.legend()
        plt.show()
    else:
        print("returning to menu")

    return clf, rf_acc_raw, rf_acc_mat, rf_class_rep
