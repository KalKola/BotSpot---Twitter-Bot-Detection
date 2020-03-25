import pandas as pd
import re


def bot_prediction(clf):

    pred_df = pd.read_csv("dataset.csv")

    print("--------------------------------------------------")
    print("                Predicting Bots                   ")
    print("--------------------------------------------------")

    bot_count = 0
    bot_list = []
    name_list = []
    for i, iter_pred in pred_df.iterrows():
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