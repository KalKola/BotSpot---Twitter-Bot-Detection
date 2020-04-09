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
from graphs import graph_display

ck = 'sRTmVkVycTYfV5G9ou34BIN5B'
ck_secret = 'KP0xxglcfkbloEA1JHBRGdjNB1m7sysqhKtMeQMjCHQBkSWqdX'
at = '1077691432360726529-6ohW6KvrlS3qlXUYvhXzqcUUmM38u0'
at_secret = '42P1J5wVfO18v5yg23uGn2XS19WzNNARw1uMft1v25q1k'


def main():

    selection = 1
    while selection != 4:
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
                print("----------------------------------")
                print("Options: ")
                print("1. Show Predicted Bots")
                print("2. Accuracy Metrics")
                print("3. Feature Importance Graph")
                print("4. Return to Main Menu")
                print("----------------------------------")

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
                    graph_display(clf, 1)

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
                print("----------------------------------")
                print("Options: ")
                print("1. Show Predicted Bots")
                print("2. Accuracy Metrics")
                print("3. Feature Distribution Graphs")
                print("4. Return to Main Menu")
                print("----------------------------------")

                svm_option = input()
                if svm_option == '1':

                    bot_prediction(clf)

                elif svm_option == '2':

                    metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

                elif svm_option == '3':
                    # Distribution Plot Display
                    graph_display(clf, 2)

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
