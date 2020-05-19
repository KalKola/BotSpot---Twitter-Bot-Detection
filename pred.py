import pandas as pd
import re


def bot_prediction(clf):

    pred_df = pd.read_csv("datasets/dataset.csv")
    pred_df = pred_df.drop("notifications", axis=1)
    pred_df = pred_df.drop("contributors_enabled", axis=1)

    print("--------------------------------------------------")
    print("                Predicting Bots                   ")
    print("--------------------------------------------------")

    # initialize bot counter for percentage determination, bot_list for account names, name_list for formatting
    bot_count = 0
    bot_list = []
    name_list = []

    # perform bot prediction on queried set
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
                               iter_pred.verified]])

        # append flagged accounts to list, format, and increment bot counter
        if is_bot == 1:
            bot_count += 1
            bot_list.append(iter_pred)
            name_format = (re.sub("'", "", iter_pred['screen_name']))
            name_list.append(name_format[1:])

    # remove duplicate accounts from the list
    name_list = list(dict.fromkeys(name_list))
    print("\n".join(name_list))
    print("Bot Percentage: " + str(bot_count/10) + "%")

    return

