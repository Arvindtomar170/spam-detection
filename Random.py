import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv("emails.csv")

df.head()

with pd.option_context('display.max_rows', 5728):
    print (df)

X, y = make_classification(n_samples=2736, n_features=2, n_informative=2, n_redundant=0, random_state=5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = RandomForestClassifier(n_estimators=20, random_state=5)

classifier.fit(X, y)
print(classifier.feature_importances_)
print(classifier.predict([[0, 0]]))

y_prediction = classifier.predict(X_test)

print(metrics.accuracy_score(y_test, y_prediction))

classifier.apply(X)


classifier.decision_path(X)

classifier.get_params(deep=True)
classifier.predict_log_proba(X)
classifier.predict_proba(X)
classifier.score(X, y)