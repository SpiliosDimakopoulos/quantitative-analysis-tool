import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

class MachineLearningAnalysis:
    """Performs machine learning analysis on the dataset."""

    @staticmethod
    def prompt_user_algorithm_choice():
        """Prompt the user to select one or more algorithms."""
        print("Please select machine learning algorithms to apply on the dataset (comma separated):")
        print("1: Logistic Regression (Classification)")
        print("2: Linear Regression (Regression)")
        print("3: Random Forest Classifier (Classification)")
        print("4: Random Forest Regressor (Regression)")
        print("5: Gradient Boosting Classifier (Classification)")
        print("6: Gradient Boosting Regressor (Regression)")
        print("7: AdaBoost Classifier (Classification)")
        print("8: Decision Tree Classifier (Classification)")
        print("9: Decision Tree Regressor (Regression)")
        print("10: Support Vector Classifier (SVC) (Classification)")
        print("11: Support Vector Regressor (SVR) (Regression)")
        print("12: K-Nearest Neighbors Classifier (Classification)")
        print("13: K-Nearest Neighbors Regressor (Regression)")
        
        attempts = 0
        while attempts < 3:
            try:
                # User input
                user_input = input("Enter your choices (comma separated, e.g., 1,2,3): ")
                choice_list = [int(x.strip()) for x in user_input.split(',')]
                
                # Validate choices
                valid_choices = set(range(1, 14))
                if all(choice in valid_choices for choice in choice_list):
                    return choice_list
                else:
                    print("Invalid choices. Please select from the available options (1-13).")
                    attempts += 1
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                attempts += 1
        
        print("Invalid input provided after 3 attempts. Defaulting to Logistic Regression.")
        return [1]  # Default to Logistic Regression

    @staticmethod
    def apply_algorithm(df, choice, target_column):
        """Apply selected algorithm on the dataset."""
        # Split the dataset into features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data (important for some models like SVM, KNN)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize dictionary for all models
        models = {
            1: LogisticRegression(random_state=42),
            2: LinearRegression(),
            3: RandomForestClassifier(n_estimators=100, random_state=42),
            4: RandomForestRegressor(n_estimators=100, random_state=42),
            5: GradientBoostingClassifier(n_estimators=100, random_state=42),
            6: GradientBoostingRegressor(n_estimators=100, random_state=42),
            7: AdaBoostClassifier(n_estimators=100, random_state=42),
            8: DecisionTreeClassifier(random_state=42),
            9: DecisionTreeRegressor(random_state=42),
            10: SVC(random_state=42),
            11: SVR(),
            12: KNeighborsClassifier(n_neighbors=5),
            13: KNeighborsRegressor(n_neighbors=5),
        }
        
        model_results = {}

        # Iterate over all the selected models
        for choice_num in choice:
            model = models[choice_num]
            if isinstance(model, LogisticRegression) or \
                isinstance(model, RandomForestClassifier) or \
                isinstance(model, GradientBoostingClassifier) or \
                isinstance(model, AdaBoostClassifier) or \
                isinstance(model, DecisionTreeClassifier) or \
                isinstance(model, SVC) or \
                isinstance(model, KNeighborsClassifier):
                model_type = "Classification"
            else:
                model_type = "Regression"

            # Fit the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model based on the type
            if model_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                model_results[choice_num] = f"Accuracy: {accuracy * 100:.2f}%"
            elif model_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                model_results[choice_num] = f"Mean Squared Error (MSE): {mse:.2f}"

        return model_results
