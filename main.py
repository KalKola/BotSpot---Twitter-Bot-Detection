import csv
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from metrics import metric_display
from pred import bot_prediction
from API import search
from random_forest import rf_pred
from svm import svm_pred

ck = 'sRTmVkVycTYfV5G9ou34BIN5B'
ck_secret = 'KP0xxglcfkbloEA1JHBRGdjNB1m7sysqhKtMeQMjCHQBkSWqdX'
at = '1077691432360726529-6ohW6KvrlS3qlXUYvhXzqcUUmM38u0'
at_secret = '42P1J5wVfO18v5yg23uGn2XS19WzNNARw1uMft1v25q1k'


def SVMPred():

    df = pd.read_csv("trainingSet.csv")
    df = df.fillna(0)

    X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
            'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
            'protected', 'verified', 'notifications', 'contributors_enabled']]
    y = df['bot']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

    # reduces computational effort by limiting the boundary space
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)

    # using preprocessing to normalize the data
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    clf = svm.SVC(kernel = 'linear', random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    clf_acc_raw = metrics.accuracy_score(y_test, y_pred)
    clf_acc_mat = confusion_matrix(y_test, y_pred)
    clf_class_rep = classification_report(y_test, y_pred)

    rfOption = 1
    while rfOption != 4:
        print("---------------")
        print("Options: ")
        print("1. Show Predicted Bots")
        print("2. Accuracy Metrics")
        print("3. Distribution Graphs")
        print("4. Return to Main Menu")
        print("---------------")

        rfOption = input()
        if rfOption == '1':

            bot_prediction(clf)

        if rfOption == '2':

            metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

        if rfOption == '3':

            # Plot of Dependent Variables
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(df['statuses_count'], df['bot'])
            plt.ylabel('Bot')
            plt.xlabel('Friends Number')
            plt.show()

            # Distribution of Dependent Variables
            plt.subplot(1, 2, 1)
            plt.hist(df['geo_enabled'][df['bot'] == 0], bins=3, alpha=0.7, label='bot = 0')
            plt.hist(df['geo_enabled'][df['bot'] == 1], bins=3, alpha=0.7, label='bot = 1')
            plt.ylabel('Distribution')
            plt.xlabel('Geo Enabled')
            plt.xticks(range(0, 2), ('No', 'Yes'))
            plt.legend()
            plt.show()

        if rfOption == '4':
             # return to main menu
            main()

def main():

    print("----------------------------")
    print("1. Fetch Tweet Data")
    print("2. RF-based Prediction")
    print("3. SVM-based Prediction")
    print("4. Exit")
    print("----------------------------")

    selection = input()

    # Twitter API call
    if selection == '1':
        search(ck, ck_secret, at, at_secret)
        print("Twitter API Fetch Complete")

    # Random Forest Model
    elif selection == '2':

        clf, clf_acc_raw, clf_acc_mat, clf_class_rep = rf_pred()

        rf_option = '1'
        while rf_option != 4:
            print("---------------")
            print("Options: ")
            print("1. Show Predicted Bots")
            print("2. Accuracy Metrics")
            print("3. Feature Importance Graph")
            print("4. Return to Main Menu")
            print("---------------")

            rf_option = input()

            # Show Predicted Bots
            if rf_option == '1':

                bot_prediction(clf)

            # Show Evaluation Metrics
            elif rf_option == '2':

                metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

            # Generate Feature Importance Graphs
            elif rf_option == '3':
                # feature importance visualization
                print("hello")

            # Return to Main Menu
            elif rf_option == '4':

                main()

            # Invalid Option Selected
            else:
                print("Invalid Input, Please Select and Option")

    # Support Vector Machine Model
    elif selection == '3':
        clf, clf_acc_raw, clf_acc_mat, clf_class_rep = svm_pred()

        svm_option = '1'
        while svm_option != 4:
            print("---------------")
            print("Options: ")
            print("1. Show Predicted Bots")
            print("2. Accuracy Metrics")
            print("3. Feature Importance Graph")
            print("4. Return to Main Menu")
            print("---------------")

            svm_option = input()
            if svm_option == '1':

                bot_prediction(clf)

            elif svm_option == '2':

                metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

            elif svm_option == '3':
                # feature importance visualization
                print("hello")

            elif svm_option == '4':
                # return to main menu
                main()
            else:
                print("Invalid Input, Please Select and Option")

    # Exit Program
    elif selection == '4':
        exit()

    # Invalid Option Selected
    else:
        print("Invalid Input")


main()
