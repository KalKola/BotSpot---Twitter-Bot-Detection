import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("trainingSet.csv")
df = df.fillna(0)

X = df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
        'protected', 'verified', 'notifications', 'contributors_enabled']]
y = df['bot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

# print("---------------------------------------------")
# print(X_train)

# reduces computational effort by limiting the boundary space

# using MinMaxScaler() to limit the boundary space
# scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_test = scaling.transform(X_test)

# using StandardScaler() to limit boundary space
# scaled = StandardScaler()
# X_train = scaled.fit_transform(X_train)
# X_test = scaled.fit_transform(X_test)

# using RobustScaler() to limit boundary space
scaled = RobustScaler()
X_train = scaled.fit_transform(X_train)
X_test = scaled.fit_transform(X_test)

# print("##############################################")
# print(X_train)
# print("---------------------------------------------")

# using preprocessing to normalize the data
# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

# generate SVM model
# clf = svm.SVC(kernel = 'linear', random_state=0)
clf = svm.SVC(kernel = 'linear', cache_size = 2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

svm_acc_raw = metrics.accuracy_score(y_test, y_pred)
svm_acc_mat = confusion_matrix(y_test, y_pred)

print(svm_acc_raw)
print(svm_acc_mat)

print("Missing Values: ")
print((df[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'default_profile', 'default_profile_image', 'geo_enabled', 'profile_background_tile',
        'protected', 'verified', 'notifications', 'contributors_enabled']] == 0).sum())

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

# Plot of Dependent Variables
plt.figure(figsize = (14,5))
plt.subplot(1,2,1)
plt.scatter(df['statuses_count'], df['bot'])
plt.ylabel('Bot')
plt.xlabel('Friends Number')
plt.show()


# Distribution of Dependent Variables
plt.subplot(1,2,1)
plt.hist(df['geo_enabled'][df['bot'] == 0], bins=3, alpha = 0.7, label = 'bot = 0')
plt.hist(df['geo_enabled'][df['bot'] == 1], bins=3, alpha = 0.7, label = 'bot = 1')
plt.ylabel('Distribution')
plt.xlabel('Geo Enabled')
plt.xticks(range(0,2), ('No', 'Yes'))
plt.legend()
plt.show()

