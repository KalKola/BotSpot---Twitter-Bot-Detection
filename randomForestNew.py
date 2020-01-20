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


#print(clf.predict([[324, 285, 16023, 210, 0]]))


print("--------------------------------------------------")
print("                Cleaning API Tweets               ")
print("--------------------------------------------------")

df = pd.read_csv("dataset.csv")

# Convert TRUE FALSE to 1 0
df['Default Profile'] = df['Default Profile'].astype(int)
df['Default Profile Image'] = df['Default Profile Image'].astype(int)
df['Geo Enabled'] = df['Geo Enabled'].astype(int)
df['Background Tile'] = df['Background Tile'].astype(int)
df['Protected'] = df['Protected'].astype(int)
df['Verified'] = df['Verified'].astype(int)
df['Notifications'] = df['Notifications'].astype(int)
df['Contributors Enabled'] = df['Contributors Enabled'].astype(int)

