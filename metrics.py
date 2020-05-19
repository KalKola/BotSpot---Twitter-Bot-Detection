import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def metric_display(raw, mat, rep):

    # create loop for metrics submenu
    met_option = 1
    while met_option != 4:
        print("__________________________________")
        print("Select an Evaluation Metric Below")
        print("1. Model Accuracy")
        print("2. Confusion Matrix Values")
        print("3. Classification Report")
        print("4. Return")
        print("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")

        met_option = input()

        # model pure-accuracy option
        if met_option == '1':

            # convert accuracy to percentage for clarity and print to screen
            print("Model Accuracy: " + str(round(raw * 100, 2)) + "%")

        # model confusion matrix option
        elif met_option == '2':

            # convert confusion matrix to seperate value and percentage
            print("Confusion Matrix Values")
            print("True Positive: " + str(mat[1][1]) + " - " + str((mat[1][1] / 2667) * 100).split('.')[0] + "%")
            print("True Negative: " + str(mat[0][0]) + " - " + str((mat[0][0]/2667)*100).split('.')[0] + "%")
            print("False Positive: " + str(mat[0][1]) + " - " + str((mat[0][1] / 2667) * 100).split('.')[0] + "%")
            print("False Negative: " + str(mat[1][0]) + " - " + str((mat[1][0]/2667)*100).split('.')[0] + "%")

            # generate confusion matrix heatmap
            fig, ax = plt.subplots()
            plt.xticks(range(0, 2), ('user', 'bot'))
            plt.yticks(range(0, 2), ('user', 'bot'))
            sns.heatmap(pd.DataFrame(mat), annot=True, cmap="Reds", fmt='g')
            ax.xaxis.set_label_position("top")
            plt.title('Model Confusion Matrix')
            plt.xlabel('Predicted Class')
            plt.ylabel('Actual Class')

            # display heatmap
            plt.show()

        # model classification report option
        elif met_option == '3':
            print("Classification Report")
            print(rep)

        # return to model submenu
        elif met_option == '4':
            return

        # invalid option selected, return to metrics submenu
        else:
            print("Invalid Selection")

