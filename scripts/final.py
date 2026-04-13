# Load data
train = pd.read_csv('/kaggle/input/competitions/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/competitions/spaceship-titanic/test.csv')

# Define Features
numerical_features = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_features = ['HomePlanet','CryoSleep','Cabin','Destination','VIP']

# Create Dataframes
X_full_uncleaned = train.drop('Transported', axis=1)
y = train['Transported']

# Impute numeric
X_num = X_full_uncleaned[numerical_features]
imp_num = SimpleImputer(strategy="median")
X_num = imp_num.fit_transform(X_num)
X_num = pd.DataFrame(X_num, columns=numerical_features, index=X_full_uncleaned.index)

# Impute categorical
X_cat = X_full_uncleaned[categorical_features]
imp_cat = SimpleImputer(strategy="most_frequent")
X_cat = imp_cat.fit_transform(X_cat)
X_cat = pd.DataFrame(X_cat, columns=categorical_features, index=X_full_uncleaned.index)

#Combine
X_full = pd.concat([X_num, X_cat], axis = 1)

# Encoding Categorical values
X_full = pd.get_dummies(X_full, columns = categorical_features, drop_first = True)

# Scaling numerical features
scaler_full = StandardScaler()
X_scale_full = X_full.copy()
X_scale_full[numerical_features] = scaler_full.fit_transform(X_full[numerical_features])

# Logistic regression
logreg = LogisticRegression(max_iter = 1000, random_state = 10)
logreg.fit(X_scale_full, y)

# TEST SET
X_uncleaned_test = test.copy()

# Keep PassengerId for submission
passenger_ids = X_uncleaned_test['PassengerId']

# Impute numeric values
X_num_test = X_uncleaned_test[numerical_features]
X_num_test = imp_num.transform(X_num_test)
X_num_test = pd.DataFrame(X_num_test, columns = numerical_features, index = X_uncleaned_test.index)

# impute categorical values
X_cat_test = X_uncleaned_test[categorical_features]
X_cat_test = imp_cat.transform(X_cat_test)
X_cat_test = pd.DataFrame(X_cat_test, columns = categorical_features, index = X_uncleaned_test.index)

X_test = pd.concat([X_num_test, X_cat_test], axis = 1)

# Encoding the categorical values
X_test = pd.get_dummies(X_test, columns = categorical_features, drop_first = True)

# Align competition columns with training columns (X_full)
X_test, X_full_aligned = X_test.align(
    X_full,
    join='right',
    axis=1,
    fill_value=0
)

# Scaling test data for numerical features
X_test_scaled = X_test.copy()
X_test_scaled[numerical_features] = scaler_full.transform(X_test_scaled[numerical_features])

# Predict for competition test set
y_pred = logreg.predict(X_test_scaled)

# Submission to the competition
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': y_pred
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")
