import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
from graphviz import Source


def graph_display(clf, deter):

    # read in training set for graph headings
    df = pd.read_csv("datasets/t_set.csv")
    df = df.fillna(0)

    X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
            'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
            'protected', 'verified']]
    y = df[['bot']]

    # RF graph option
    if deter == 1:

        # calculate RF feature importance
        f_imp = clf.feature_importances_

        # sort the array of features by importance
        sorted_ind = np.argsort(f_imp)[::-1]
        feature_list = list(X.columns)

        print("----------------------------------")
        print("        Feature Importance        ")
        print("----------------------------------")

        # print sorted feature list importance level
        for x in range(X.shape[1]):
            temp = sorted_ind[x]
            print("%d. " % (x + 1) + feature_list[temp] + "-- %.2f" % round((f_imp[temp] * 100), 2) + "%")

        # building bar-chart for RF f-imp
        # plt.style.use('fivethirtyeight')
        # plt.style.use('bmh')
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 10))
        x_values = list(range(len(f_imp)))
        plt.bar(x_values, f_imp, color="g", orientation='vertical')
        plt.xticks(x_values, feature_list, rotation=25)
        plt.ylabel('Importance'); plt.xlabel('Features'); plt.title('Feature Importance')

        figure = plt.gcf()
        plt.show()

        # prompt user to save figure
        save_fig = 1
        fig_name = 0
        print("Save to local machine?")
        print(" 1. Yes")
        print(" 2. No")

        save_fig = input()
        if save_fig == '1':
            print("Enter File Name:")
            fig_name = input()

            print("Saving figure to local machine...")
            figure.savefig("graphs/" + fig_name + '.png', dpi=100)
            print("Save complete")
        else:
            print("Figure not saved")

        # Decision Tree Visualization
        rf_estimator = clf.estimators_[5]
        target_feature = ["bot", "genuine"]

        # Create DOT file for Random Forest Decision Tree
        export_graphviz(rf_estimator,
                        out_file='decision_tree.dot',
                        feature_names=X.columns,
                        class_names=target_feature,
                        proportion=True,
                        rounded=True,
                        precision=2,
                        filled=True)

        # Display Results as a PNG file
        # s = Source.from_file('decision_tree.dot')
        # s.view()

        return

    # SVM, LR graph option
    elif deter == 2:

        # initialize graph_list for menu option selection
        graph_list = ['geo_enabled', 'default_profile', 'default_profile_image',
                      'favourites_count', 'statuses_count', 'followers_count',
                      'friends_count', 'listed_count']

        # create loop for plot generation
        graph_opt = 1
        while graph_opt != 9:
            print("----------------------------")
            print("1. Distribution - Geo Enabled")
            print("2. Distribution - Default Profile")
            print("3. Distribution - Default Image")
            print("4. Scatter Plot - Favourites Count")
            print("5. Scatter Plot - Statuses Count")
            print("6. Scatter Plot - Followers Count")
            print("7. Scatter Plot - Friends Count")
            print("8. Scatter Plot - Listed Count")
            print("9. Exit")
            print("----------------------------")

            graph_opt = input()

            # distribution plot options
            if graph_opt in ('1', '2', '3'):

                # Distribution of Independent Variables
                plt.subplot(1, 2, 1)
                plt.hist(df[graph_list[int(graph_opt) - 1]][df['bot'] == 0], bins=3, alpha=0.5, label='bot = 0')
                plt.hist(df[graph_list[int(graph_opt) - 1]][df['bot'] == 1], bins=3, alpha=0.5, label='bot = 1')
                plt.ylabel('Distribution')
                plt.xlabel(graph_list[int(graph_opt) - 1])
                plt.xticks(range(0, 2), ('No', 'Yes'))
                plt.legend()

                figure = plt.gcf()
                plt.show()

                save_fig = 1
                fig_name = 0
                print("Save to local machine?")
                print(" 1. Yes")
                print(" 2. No")

                save_fig = input()
                if save_fig == '1':
                    print("Enter File Name:")
                    fig_name = input()

                    print("Saving figure to local machine...")
                    figure.savefig("graphs/" + fig_name + '.png', dpi=100)
                    print("Save complete")
                else:
                    print("Figure not saved")

            # scatter plot options
            elif graph_opt in ('4', '5', '6', '7', '8'):

                # Plot of Independent Variables
                plt.figure(figsize=(14, 5))
                plt.subplot(1, 2, 1)
                plt.scatter(df[graph_list[int(graph_opt) - 1]], df['bot'])
                plt.ylabel('Bot')
                plt.xlabel(graph_list[int(graph_opt) - 1])

                figure = plt.gcf()
                plt.show()

                save_fig = 1
                fig_name = 0
                print("Save to local machine?")
                print(" 1. Yes")
                print(" 2. No")

                save_fig = input()
                if save_fig == '1':
                    print("Enter File Name:")
                    fig_name = input()

                    print("Saving figure to local machine...")
                    figure.savefig("graphs/" + fig_name + '.png', dpi=100)
                    print("Save complete")
                else:
                    print("Figure not saved")

            # return to model submenu
            elif graph_opt == '9':

                return

            # invalid option selected, return to graph submenu
            else:

                print("Invalid Option, Please Select a Feature to Graph")

