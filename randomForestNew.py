import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("trainingSet.csv")
df = df.fillna(0)

df.head()


# print("--------------------------------------")
# print(df['following'].describe())
# print("--------------------------------------")

# For Feature Importance Graph
fImpGraph = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
        'protected', 'verified', 'notifications', 'contributors_enabled']

X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
        'protected', 'verified', 'notifications', 'contributors_enabled']]
y = df['bot']

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# initializing the RF model with optimized parameters - rfTest()
clf = RandomForestClassifier(n_estimators=1000,
                             max_depth=20,
                             max_features='auto',
                             oob_score='TRUE',
                             bootstrap='TRUE',
                             random_state=10)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# generating accuracy metrics, confusion matrix, and classification report
rf_acc_raw = metrics.accuracy_score(y_test, y_pred)
rf_acc_mat = confusion_matrix(y_test, y_pred)
rf_class_rep = classification_report(y_test, y_pred)

print(rf_acc_raw)
print(rf_acc_mat)
print(rf_class_rep)

# print("--------------------------------------------------")
# print("                Cleaning API Tweets               ")
# print("--------------------------------------------------")

df2 = pd.read_csv("dataset.csv")

# Convert TRUE FALSE to 1 0
# df2['default_profile'] = df2['default_profile'].astype(int)
# df2['default_profile_image'] = df2['default_profile_image'].astype(int)
# df2['geo_enabled'] = df2['geo_enabled'].astype(int)
# df2['profile_background_tile'] = df2['profile_background_tile'].astype(int)
# df2['protected'] = df2['protected'].astype(int)
# df2['verified'] = df2['verified'].astype(int)
# df2['notifications'] = df2['notifications'].astype(int)
# df2['contributors_enabled'] = df2['contributors_enabled'].astype(int)

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
        print(j['screen_name'])

print("Bot Count: " + str(botCount))


# feature importance visualization
f_imp = clf.feature_importances_

# compute the standard deviation for each feature
#std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

#sort the array of features by importance
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
#plt.xlim([-1, X.shape[1]])
plt.show()