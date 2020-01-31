from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("trainingSet.csv")
df = df.fillna(0)

df.head()

# statuses_count
# followers_count
# friends_count
# favourites_count
# listed_count
# default_profile
# default_profile_image
# geo_enabled
# profile_background_tile
# protected
# verified
# notifications
# contributors_enabled

# name
# screen_name
# time_zone (108 unique)
# location (2109 unique)
# profile_text_color (405 unique)
# profile_background_color (531 unique)
# profile_link_color (895 unique)
# description
# following
# created_at

print("--------------------------------------")
print(df['following'].describe())
print("--------------------------------------")

X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
        'protected', 'verified', 'notifications', 'contributors_enabled']]
y = df['bot']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# MODEL EVALUATION
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


print("--------------------------------------------------")
print("                Cleaning API Tweets               ")
print("--------------------------------------------------")

df2 = pd.read_csv("dataset.csv")

# Convert TRUE FALSE to 1 0
df2['default_profile'] = df2['default_profile'].astype(int)
df2['default_profile_image'] = df2['default_profile_image'].astype(int)
df2['geo_enabled'] = df2['geo_enabled'].astype(int)
df2['profile_background_tile'] = df2['profile_background_tile'].astype(int)
df2['protected'] = df2['protected'].astype(int)
df2['verified'] = df2['verified'].astype(int)
df2['notifications'] = df2['notifications'].astype(int)
df2['contributors_enabled'] = df2['contributors_enabled'].astype(int)

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
        botCount += 1

print("Bot Count: " + str(botCount))