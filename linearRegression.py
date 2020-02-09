import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

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

clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


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

# for i, j in df2.iterrows():
#    if j.screen_name == "b'90210Deplorable'":
#        x0 = j.screen_name
#        x1 = j.statuses_count
#        x2 = j.followers_count
#        x3 = j.friends_count
#        x4 = j.favourites_count
#        x5 = j.listed_count
#        x6 = j.default_profile
#        x7 = j.default_profile_image
#        x8 = j.geo_enabled
#        x9 = j.profile_background_tile
#        x10 = j.protected
#        x11 = j.verified
#        x12 = j.notifications
#        x13 = j.contributors_enabled

#print(x0)
#print(clf.predict([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]]))