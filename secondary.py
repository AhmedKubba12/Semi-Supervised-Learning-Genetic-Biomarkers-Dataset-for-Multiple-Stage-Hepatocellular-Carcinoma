from sklearn.semi_supervised import LabelSpreading
import pandas as pd

combined_dataset = pd.read_csv('Combined_Dataset.csv')
large_dataset = pd.read_csv('TCGA_Dataset.csv')
small_dataset = pd.read_csv('Lubeck.csv')

true_labels = y_labeled = small_dataset['symbol']
num_extended_rows = 782
extended_labels = pd.Series([-1] * num_extended_rows)
true_labels = pd.concat([true_labels, extended_labels], ignore_index=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_labels = le.fit_transform(true_labels[:28])
true_labels[:28] = encoded_labels

true_labels=true_labels.astype('int')

print(combined_dataset)
print(true_labels)

lp_model = LabelSpreading(max_iter=2000)
lp_model.fit(combined_dataset, true_labels)
predicted_labels = lp_model.transduction_

print(predicted_labels)

print(le.inverse_transform(predicted_labels))