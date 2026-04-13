# Fitting on training data set

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

numeric_features = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_features = ['HomePlanet','CryoSleep','Cabin','Destination','VIP']

X_uncleaned = train.drop('Transported', axis=1)
y = train['Transported']

# Impute numeric
X_num = X_uncleaned[numeric_features]
imp_num = SimpleImputer(strategy="median")
X_num = imp_num.fit_transform(X_num)
X_num = pd.DataFrame(X_num, columns=numeric_features, index=X_uncleaned.index)

# Impute categorical
X_cat = X_uncleaned[categorical_features]
imp_cat = SimpleImputer(strategy="most_frequent")
X_cat = imp_cat.fit_transform(X_cat)
X_cat = pd.DataFrame(X_cat, columns=categorical_features, index=X_uncleaned.index)

# Combine both
X = pd.concat([X_num, X_cat], axis=1)

# Now encode categoricals
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10, stratify=y)
logreg = LogisticRegression(max_iter=1000, random_state = 10)

# Scale numeric features only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Overwrite the original names at the end
X_train = X_train_scaled
X_test = X_test_scaled

# Fitting the data
logreg.fit(X_train_scaled, y_train)

# Predicting the values
y_pred = logreg.predict(X_test_scaled)

# Accuracy
score = accuracy_score(y_test, y_pred)
print("Validation Score:", score)
