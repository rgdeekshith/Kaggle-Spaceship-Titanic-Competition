import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# Load data
# =========================
train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test  = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')

# =========================
# TRAIN + VALIDATION (your pattern)
# =========================

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.3,
    random_state = 10,
    stratify=y
)

logreg = LogisticRegression(max_iter=1000, random_state = 10)

# Scale numeric features only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features]  = scaler.transform(X_test[numeric_features])

# Overwrite the original names at the end
X_train = X_train_scaled
X_test  = X_test_scaled

# Fitting the data
logreg.fit(X_train, y_train)

# Predicting the values
y_pred = logreg.predict(X_test)

# Accuracy
score = accuracy_score(y_test, y_pred)
print("Validation accuracy:", score)

# =========================
# REFIT MODEL ON FULL TRAIN
# =========================

# Rebuild X on ALL train rows using same imputers and encoding

X_uncleaned_full = train.drop('Transported', axis=1)

# Impute numeric with imp_num (already fitted)
X_num_full = X_uncleaned_full[numeric_features]
X_num_full = imp_num.transform(X_num_full)
X_num_full = pd.DataFrame(X_num_full,
                          columns=numeric_features,
                          index=X_uncleaned_full.index)

# Impute categorical with imp_cat (already fitted)
X_cat_full = X_uncleaned_full[categorical_features]
X_cat_full = imp_cat.transform(X_cat_full)
X_cat_full = pd.DataFrame(X_cat_full,
                          columns=categorical_features,
                          index=X_uncleaned_full.index)

# Combine
X_full = pd.concat([X_num_full, X_cat_full], axis=1)

# One‑hot encode categoricals
X_full = pd.get_dummies(X_full, columns=categorical_features, drop_first=True)

# Scale numeric on full train with a new scaler
scaler_full = StandardScaler()
X_full_scaled = X_full.copy()
X_full_scaled[numeric_features] = scaler_full.fit_transform(X_full[numeric_features])

# Final logistic regression on ALL train data
logreg_final = LogisticRegression(max_iter=1000, random_state=10)
logreg_final.fit(X_full_scaled, y)

# =========================
# APPLY SAME STEPS TO TEST.CSV
# =========================

# Start from raw test
X_uncleaned_comp = test.copy()

# Keep PassengerId for submission
passenger_ids = X_uncleaned_comp['PassengerId']

# Impute numeric on competition test
X_num_comp = X_uncleaned_comp[numeric_features]
X_num_comp = imp_num.transform(X_num_comp)
X_num_comp = pd.DataFrame(X_num_comp,
                          columns=numeric_features,
                          index=X_uncleaned_comp.index)

# Impute categorical on competition test
X_cat_comp = X_uncleaned_comp[categorical_features]
X_cat_comp = imp_cat.transform(X_cat_comp)
X_cat_comp = pd.DataFrame(X_cat_comp,
                          columns=categorical_features,
                          index=X_uncleaned_comp.index)

# Combine
X_comp = pd.concat([X_num_comp, X_cat_comp], axis=1)

# One‑hot encode categoricals on competition test
X_comp = pd.get_dummies(X_comp, columns=categorical_features, drop_first=True)

# Align competition columns with training columns (X_full)
X_comp, X_full_aligned = X_comp.align(
    X_full,
    join='right',
    axis=1,
    fill_value=0
)

# Scale numeric features in competition set using scaler_full
X_comp_scaled = X_comp.copy()
X_comp_scaled[numeric_features] = scaler_full.transform(X_comp[numeric_features])

# Predict for competition test set
y_comp_pred = logreg_final.predict(X_comp_scaled)

# =========================
# BUILD SUBMISSION
# =========================

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': y_comp_pred
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")
