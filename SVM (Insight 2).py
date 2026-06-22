from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Importing data
stud_performance = pd.read_csv('modif_student_performance.csv')

# Define features based on your selected insights
features = [col for col in stud_performance.columns if ('_encoded' in col or '_binned' in col) and col != 'Motivation_Level_encoded_o']
target = 'Motivation_Level_encoded_o'

X = stud_performance[features]
y = stud_performance[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(stud_performance.corr(numeric_only=True)['Motivation_Level_encoded_o'].sort_values(ascending=False)) # Display correlation with target variable

# Create and train the SVM model
print("Training SVM model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n\nF1 Score:", round(f1_score(y_test, y_pred, average='weighted') * 100, 2), "%")
print("\nPrecision Score:", round(precision_score(y_test, y_pred, average='weighted') * 100, 2), "%")
print("\nRecall Score:", round(recall_score(y_test, y_pred, average='weighted') * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Display the distribution of the target variable
print("Distribution of target variable:")
print(y.value_counts().sort_index())

# Check correlation with target variable
print("\nCorrelation with target variable:")
print(stud_performance.corr(numeric_only=True)['Motivation_Level_encoded_o'].sort_values(ascending=False))

# Save the model if needed
# import joblib
# joblib.dump(svm_model, 'svm_model.pkl')
