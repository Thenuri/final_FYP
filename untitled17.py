from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
}

# Create a Gradient Boosting Classifier
gradient_boosting_model = GradientBoostingClassifier(random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(
    estimator=gradient_boosting_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # Use 5-fold cross-validation
    n_jobs=-1  # Use all available CPU cores
)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# Print the best score (accuracy) found
print("Best Score:", grid_search.best_score_)

# Get the best estimator (model)
best_gradient_boosting_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_gradient_boosting_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Print the classification report
report = classification_report(
    y_test, y_pred,
    target_names=le_outfit_category.classes_,
    zero_division=0
)
print("Classification Report:")
print(report)