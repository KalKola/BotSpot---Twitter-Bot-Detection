from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv("TestData.csv")
df = df.fillna(0)

df.head()

X = df[['favourites_count', 'followers_count', 'statuses_count', 'friends_count', 'default_profile']]
y = df['Bot']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

print(clf.predict([[324, 285, 16023, 210, 0]]))





