import csv
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import re

from metrics import metric_display

ck = 'sRTmVkVycTYfV5G9ou34BIN5B'
ck_secret = 'KP0xxglcfkbloEA1JHBRGdjNB1m7sysqhKtMeQMjCHQBkSWqdX'
at = '1077691432360726529-6ohW6KvrlS3qlXUYvhXzqcUUmM38u0'
at_secret = '42P1J5wVfO18v5yg23uGn2XS19WzNNARw1uMft1v25q1k'

def fTweet():
    print("Enter Hashtag Below: ")
    hashtag = input()

    #using access tokens and consumer keys to set up the API
    print("Connecting to Twitter API")
    auth = tweepy.OAuthHandler(ck, ck_secret)
    auth.set_access_token(at, at_secret)

    # wait_on_rate_limit prevents the module from exceeding the API limit
    api = tweepy.API(auth, wait_on_rate_limit=True)

    print("Fetching Tweets from " + hashtag)

    #create the csv file to write the data to
    fname = 'dataset'
    with open('%s.csv' % (fname), 'w') as file:

            fileWriter = csv.writer(file)

            fileWriter.writerow(['name', 'screen_name', 'statuses_count', 'followers_count', 'friends_count',
                                 'favourites_count', 'listed_count', 'time_zone', 'location', 'default_profile',
                                 'default_profile_image', 'geo_enabled', 'profile_text_color', 'profile_background_tile',
                                 'profile_background_color', 'profile_link_color', 'protected', 'verified', 'notifications',
                                 'description', 'contributors_enabled', 'following', 'created_at'])

            print("Titles Written")

            for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en", tweet_mode='extended').items(1000):

                fileWriter.writerow([
                            tweet.user.name.encode('utf-8'),
                            tweet.user.screen_name.encode('utf-8'),

                            tweet.user.statuses_count,
                            tweet.user.followers_count,
                            tweet.user.friends_count,
                            tweet.user.favourites_count,
                            tweet.user.listed_count,

                            tweet.user.time_zone,
                            tweet.user.location.encode('utf-8'),
                            int(tweet.user.default_profile),
                            int(tweet.user.default_profile_image),
                            int(tweet.user.geo_enabled),
                            tweet.user.profile_text_color,
                            int(tweet.user.profile_background_tile),
                            tweet.user.profile_background_color,
                            tweet.user.profile_link_color,

                            int(tweet.user.protected),
                            int(tweet.user.verified),
                            int(tweet.user.notifications),

                            tweet.user.description.encode('utf-8'),
                            int(tweet.user.contributors_enabled),
                            tweet.user.following,
                            tweet.user.created_at
                            ])

    # return to main menu
    main()

def RFPred():
    df = pd.read_csv("trainingSet.csv")
    df = df.fillna(0)

    fImpGraph = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
                 'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
                 'protected', 'verified', 'notifications', 'contributors_enabled']

    X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
            'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
            'protected', 'verified', 'notifications', 'contributors_enabled']]
    y = df['bot']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    clf = RandomForestClassifier(n_estimators=100, random_state=10)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    clf_acc_raw = metrics.accuracy_score(y_test, y_pred)
    clf_acc_mat = confusion_matrix(y_test, y_pred)
    clf_class_rep = classification_report(y_test, y_pred)

    df2 = pd.read_csv("dataset.csv")


    rfOption = 1
    while rfOption != 4:
        print("---------------")
        print("Options: ")
        print("1. Show Predicted Bots")
        print("2. Accuracy Metrics")
        print("3. Feature Importance Graph")
        print("4. Return to Main Menu")
        print("---------------")

        rfOption = input()
        if rfOption == '1':
            print("--------------------------------------------------")
            print("                Predicting Bots                   ")
            print("--------------------------------------------------")

            bot_count = 0
            bot_list = []
            name_list = []
            for i, iter_pred in df2.iterrows():
                is_bot = clf.predict([[iter_pred.statuses_count,
                                       iter_pred.followers_count,
                                       iter_pred.friends_count,
                                       iter_pred.favourites_count,
                                       iter_pred.listed_count,
                                       iter_pred.default_profile,
                                       iter_pred.default_profile_image,
                                       iter_pred.geo_enabled,
                                       iter_pred.profile_background_tile,
                                       iter_pred.protected,
                                       iter_pred.verified,
                                       iter_pred.notifications,
                                       iter_pred.contributors_enabled]])

                if is_bot == 1:
                    bot_count += 1
                    bot_list.append(iter_pred)
                    name_format = (re.sub("'", "", iter_pred['screen_name']))
                    name_list.append(name_format[1:])

            print("\n".join(name_list))
            print("Bot Count: " + str(bot_count))

        elif rfOption == '2':

                metric_display(clf_acc_raw, clf_acc_mat, clf_class_rep)

        elif rfOption == '3':
            # feature importance visualization
            f_imp = clf.feature_importances_

            # compute the standard deviation for each feature
            # std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

            # sort the array of features by importance
            indices = np.argsort(f_imp)[::-1]

            print("-------------------")
            print("Feature Importance")
            print("-------------------")

            for x in range(X.shape[1]):
                temp = indices[x]
                print("%d." % (x + 1) + fImpGraph[temp] + "(%f)" % (f_imp[temp]))

            # plot feature importance on graph
            plt.figure()
            plt.title("Random Forest Feature Importance")
            plt.bar(range(X.shape[1]), f_imp[indices], color="g", align="center")
            plt.xticks(range(X.shape[1]), indices)
            # plt.xlim([-1, X.shape[1]])
            plt.show()

        elif rfOption == '4':
            # return to main menu
            main()
        else:
            print("Invalid Input, Please Select and Option")

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

            df2 = pd.read_csv("dataset.csv")

            print("--------------------------------------------------")
            print("                Predicting Bots                   ")
            print("--------------------------------------------------")

            botCount = 0
            for i, j in df2.iterrows():
                isBot = clf.predict([[j.statuses_count, j.followers_count, j.friends_count, j.favourites_count,
                                      j.listed_count, j.default_profile, j.default_profile_image, j.geo_enabled,
                                      j.profile_background_tile, j.protected, j.verified, j.notifications,
                                      j.contributors_enabled]])

                if isBot == 1:
                    # Save Bot Results for Printing
                    print(j.screen_name)
                    botCount += 1

            print("Bot Count: " + str(botCount))

        if rfOption == '2':

            print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))

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

    if selection == '1':
        fTweet()
        print("Twitter API Fetch Complete")
    elif selection == '2':
        RFPred()
    elif selection == '3':
        SVMPred()
    elif selection == '4':
        exit()
    else:
        print("Invalid Input")


main()