import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def metric_display(raw, mat, rep):
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
        print(met_option)
        if met_option == '1':
            print("Model Accuracy: " + str(round(raw * 100, 2)) + "%")
        elif met_option == '2':
            print("Confusion Matrix Values")
            print("True Positive: " + str(mat[1][1]) + " - " + str((mat[1][1] / 2667) * 100).split('.')[0] + "%")
            print("True Negative: " + str(mat[0][0]) + " - " + str((mat[0][0]/2667)*100).split('.')[0] + "%")
            print("False Positive: " + str(mat[0][1]) + " - " + str((mat[0][1] / 2667) * 100).split('.')[0] + "%")
            print("False Negative: " + str(mat[1][0]) + " - " + str((mat[1][0]/2667)*100).split('.')[0] + "%")

            class_names = [0, 1]
            fig, ax = plt.subplots()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
            sns.heatmap(pd.DataFrame(mat), annot=True, cmap="YlGnBu", fmt='g')
            ax.xaxis.set_label_position("top")
            plt.tight_layout()
            plt.title('Confusion Matrix', y=1.1)
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')

            plt.show()

        elif met_option == '3':
            print("Classification Report")
            print(rep)
        elif met_option == '4':
            return
        else:
            print("Invalid Selection")

    return
