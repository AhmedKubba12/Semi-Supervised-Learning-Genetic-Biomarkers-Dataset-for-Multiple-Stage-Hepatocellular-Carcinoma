import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

small_dataset = pd.read_csv('Lubeck2.csv')

true_labels = small_dataset['symbol']
small_dataset = small_dataset.drop(columns=['symbol'])

print(true_labels)
print(small_dataset)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_labels = le.fit_transform(true_labels)
true_labels = encoded_labels

print(encoded_labels)

X_train, X_test, y_train_true, y_test_true = train_test_split(small_dataset, true_labels, test_size=0.3, random_state=42, stratify=true_labels)

model = XGBClassifier(objective='multi:softproba', num_class=5, random_state=42)

model.fit(X_train, y_train_true)

y_test_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_true, y_test_pred)
precision = precision_score(y_test_true, y_test_pred, average='macro')
recall = recall_score(y_test_true, y_test_pred, average='macro')
f1 = f1_score(y_test_true, y_test_pred, average='macro')

from sklearn.metrics import log_loss
# Calculate cross-entropy loss
y_test_prob = model.predict_proba(X_test)  # Probabilities for each class
cross_entropy_loss = log_loss(y_test_true, y_test_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Cross-Entropy Loss:", cross_entropy_loss)

conf_matrix = confusion_matrix(y_test_true, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

