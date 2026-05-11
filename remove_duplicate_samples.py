import pandas as pd

# Read the CSV file
original_dataset = pd.read_csv('Lubeck5 (on Lubeck4 dataset).csv')

# Remove duplicate rows based on all columns
deduplicated_dataset = original_dataset.drop_duplicates()

# Save the deduplicated dataset to a new CSV file
deduplicated_dataset.to_csv('Lubeck5 (on Lubeck4 dataset).csv', index=False)
