from metrics import metric_display
from pred import bot_prediction
from API import search
from random_forest import rf_pred
from svm import svm_pred
from logistic_regression import lr_pred
from graphs import graph_display

ck = ''
ck_secret = ''
at = ''
at_secret = ''

def main():

    # create loop for main menu
    selection = 1
    while selection != 4:
        print("----------------------------")
        print("1. Fetch Tweet Data")
        print("2. RF-based Prediction")
        print("3. SVM-based Prediction")
        print("4. Logistic Regression Prediction")
        print("5. Exit")
        print("----------------------------")

        selection = input()

        # call Twitter API module
        if selection == '1':
            search(ck, ck_secret, at, at_secret)
            print("Twitter API Fetch Complete")

        # call Random Forest module
        elif selection == '2':

            # accept model, accuracy metrics from RF module
            clf, clf_acc_raw, clf_acc_mat, clf_class_rep = rf_pred()

            # create loop for RF submenu
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

                # bot prediction option
                if rf_option == '1':

                    # call pred module, pass classifier
                    bot_prediction(clf)

                # evaluation metrics option
                elif rf_option == '2':

                    # call metrics module, pass model metrics
                    metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

                # visualization option
                elif rf_option == '3':

                    # call graph module for feature importance graph
                    graph_display(clf, 1)

                # return to main menu
                elif rf_option == '4':

                    main()

                # invalid option selected, return to submenu
                else:
                    print("Invalid Input, Please Select and Option")

        # call Support Vector Machine module
        elif selection == '3':

            # accept model, accuracy metrics from SVM module
            clf, clf_acc_raw, clf_acc_mat, clf_class_rep = svm_pred()

            # create loop for SVM submenu
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

                # bot prediction option
                if svm_option == '1':

                    # call pred module, pass classifier
                    bot_prediction(clf)

                # evaluation metrics option
                elif svm_option == '2':

                    # call metrics module, pass model metrics
                    metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

                # visualization option
                elif svm_option == '3':

                    # call graph module to create feature plots
                    graph_display(clf, 2)

                # return to main menu
                elif svm_option == '4':

                    main()

                # invalid option selected, return to submenu
                else:
                    print("Invalid Input, Please Select and Option")

        # call Logistic Regression Module
        elif selection == '4':

            # accept model, accuracy metrics from LR module
            clf, clf_acc_raw, clf_acc_mat, clf_class_rep = lr_pred()

            # create loop for LR submenu
            lr_option = '1'
            while lr_option != 4:
                print("----------------------------------")
                print("Options: ")
                print("1. Show Predicted Bots")
                print("2. Accuracy Metrics")
                print("3. Feature Distribution Graphs")
                print("4. Return to Main Menu")
                print("----------------------------------")

                lr_option = input()

                # bot prediction option
                if lr_option == '1':

                    # call prediction module, pass in classifier
                    bot_prediction(clf)

                # evaluation metrics option
                elif lr_option == '2':

                    # call metrics module, pass in model metrics
                    metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

                # visualization module option
                elif lr_option == '3':

                    # call graph module to create feature plots
                    graph_display(clf, 2)

                # return to main menu
                elif lr_option == '4':

                    main()

                # invalid option selected, return to submenu
                else:
                    print("Invalid Input, Please Select and Option")

        # exit system
        elif selection == '5':
            exit()

        # Invalid Option Selected
        else:
            print("Invalid Input")


main()
