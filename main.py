import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load datasets
small_dataset = pd.read_csv('Lubeck.csv')
large_dataset = pd.read_csv('TCGA_Dataset.csv')

# Identify common genes
commonGenes = list(set(large_dataset.columns) & set(small_dataset.columns))

# Select relevant columns
large_dataset = large_dataset[['Sample_type'] + commonGenes]
small_dataset = small_dataset[['symbol'] + commonGenes]

combined_dataset = pd.concat([small_dataset, large_dataset], ignore_index=True)

true_labels = small_dataset['symbol']
small_dataset = small_dataset.drop(columns=['symbol'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_labels = le.fit_transform(true_labels)
true_labels = encoded_labels

# X_combined = combined_dataset.drop(columns=['symbol', 'Sample_type'])
# y_combined = combined_dataset['symbol']

# print(X_combined)
# print(y_combined)
# X_train, X_test, y_train_true, y_test_true = train_test_split(small_dataset, true_labels, test_size=0.2, random_state=42)

# Train Random Forest Classifier on the combined dataset
model = XGBClassifier(objective='multi:softproba', num_class=5, random_state=42)
model.fit(small_dataset, true_labels)

# Make predictions on the large dataset
X_large = large_dataset.drop(columns=['Sample_type'])
y_large_true = large_dataset['Sample_type']
y_large_pred = le.inverse_transform(model.predict(X_large))

# print(small_dataset)
# print(true_labels)
import numpy as np
for i in range(1):
    y_large_pred_prob = model.predict_proba(X_large)
    selected_indices = [index for index, inner_list in enumerate(y_large_pred_prob) if max(inner_list) >= 0.7]

    for index in selected_indices:
        row_to_add1 = pd.DataFrame(X_large.iloc[index]).transpose()
        small_dataset = pd.concat([small_dataset, row_to_add1], ignore_index=True)

        row_to_add2 = list(y_large_pred[index])
        print(row_to_add2)
        true_labels = le.inverse_transform(true_labels)
        print(true_labels)
        true_labels = np.concatenate((true_labels, row_to_add2))
        print(row_to_add2)
        print(true_labels)
        # true_labels = pd.concat([true_labels, row_to_add2], ignore_index=True)
        # true_labels = le.fit_transform(true_labels)
    print(small_dataset)
    print(le.inverse_transform(true_labels))
    '''
    model.fit(small_dataset, true_labels)

    X_large = large_dataset.drop(columns=['Sample_type'])
    y_large_true = large_dataset['Sample_type']
    y_large_pred = le.inverse_transform(model.predict(X_large))

    print(y_combined)



# Create a DataFrame for results
transposed_data = list(map(list, zip(*y_large_pred_prob)))
result_df = pd.DataFrame({'True_Labels': y_large_true,
                          'Predicted_Labels': y_large_pred,
                          'ehcc': transposed_data[0],
                          'hgdn': transposed_data[1],
                          'lgdn': transposed_data[2],
                          'phcc': transposed_data[3],
                          'sl': transposed_data[4]})
result_df.to_csv('semi_supervised_results000.csv', index=False)
'''