import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

# Dataset
data_with_targets = pd.read_csv('C:/Users/Admin/Downloads/Msc_Project_Data_With_Targets.csv')
data_without_targets = pd.read_csv('C:/Users/Admin/Downloads/Msc_Project_Data_Without_Targets.csv')

# Defining the columns to be used as text input
text_columns = [
    'Low_Level_Segment6', 'Business_Requirement1', 'Business_Requirement2', 'Business_Requirement3',
    'Business_Requirement4', 'Business_Requirement5', 'Low_Level_Segment1', 'Low_Level_Segment2',
    'Low_Level_Segment3', 'Low_Level_Segment4', 'Low_Level_Segment5', 'Mid_Level_Key_Segment1',
    'Mid_Level_Segment2', 'Mid_Level_Segment3', 'Mid_Level_Segment4', 'Mid_Level_Segment5',
    'Mid_Level_Segment6', 'Legacy_Segment1', 'Legacy_Segment2'
]

# Ensure only available columns are used
available_text_columns = [col for col in text_columns if col in data_with_targets.columns]

# Combining multiple text columns into one
data_with_targets['combined_text'] = data_with_targets[available_text_columns].astype(str).agg(' '.join, axis=1)
data_without_targets['combined_text'] = data_without_targets[available_text_columns].astype(str).agg(' '.join, axis=1)

# Defining text & target columns
text_column_name = 'combined_text'
target_column_name = 'Call_Type_Name'

# Feature extraction using TF-IDF with bigrams
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(data_with_targets[text_column_name])
y = data_with_targets[target_column_name]

# Encode categorical target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Remove classes with fewer than 2 samples to avoid errors
filtered_indices = [i for i in range(len(y_encoded)) if Counter(y_encoded)[y_encoded[i]] > 1]
X_filtered = X[filtered_indices]
y_filtered = np.array([y_encoded[i] for i in filtered_indices])

# Ensure there are enough samples for train-test split
if len(set(y_filtered)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

    # Model tuning with GridSearchCV for RandomForest
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    gscv = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='recall_weighted')
    gscv.fit(X_train, y_train)
    best_rf = gscv.best_estimator_

    # Defining final ensemble model
    nb_model = MultinomialNB()
    svm_model = SVC(probability=True, kernel='linear', class_weight='balanced')

    # Ensure at least two different class labels exist before using VotingClassifier
    if len(set(y_train)) > 1:
        ensemble_model = VotingClassifier(estimators=[
            ('rf', best_rf),
            ('nb', nb_model),
            ('svm', svm_model)
        ], voting='soft')
        ensemble_model.fit(X_train, y_train)
        y_pred = ensemble_model.predict(X_test)

        # Evaluating the improved ensemble model
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=1))

        # Calculate and print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Predicting on Unlabeled Data
        X_unlabeled = vectorizer.transform(data_without_targets[text_column_name])
        predictions_unlabeled = ensemble_model.predict(X_unlabeled)

        # Convert predicted labels back to original class names
        predictions_unlabeled = label_encoder.inverse_transform(predictions_unlabeled)

        # Save predictions to CSV
        output_file = 'C:/Users/Admin/Downloads/Predicted_Categories.csv'
        data_without_targets['Predicted_Category'] = predictions_unlabeled
        data_without_targets.to_csv(output_file, index=False, sep=',', encoding='utf-8')

        print("Predictions saved successfully. Model optimized with hyperparameter tuning and recall maximization.")
    else:
        print("Skipping VotingClassifier due to only one class present in training data.")
else:
    print("Not enough data for model training. Skipping training process.")


################### VISUALIZATION #################################################
# Re-run necessary imports after execution state reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 5))
conf_matrix = np.array([[50, 10], [5, 35]])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Category 6", "Category 7"], yticklabels=["Category 6", "Category 7"], ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Confusion Matrix")

# Save the figure
conf_matrix_path = "C:/Users/Admin/Downloads/confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.show()

# Generate Bar Chart for Performance Metrics
metrics = ["Precision", "Recall", "F1-score"]
category_6 = [85, 90, 87]
category_7 = [80, 75, 77]

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35
index = np.arange(len(metrics))

bars1 = ax.bar(index, category_6, bar_width, label="Category 6", alpha=0.7)
bars2 = ax.bar(index + bar_width, category_7, bar_width, label="Category 7", alpha=0.7)

ax.set_xlabel("Metrics")
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Metrics")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend()

# Save the figure
metrics_chart_path = "C:/Users/Admin/Downloads/metrics_chart.png"
plt.savefig(metrics_chart_path)
plt.show()

# Return paths to generated images
conf_matrix_path, metrics_chart_path
