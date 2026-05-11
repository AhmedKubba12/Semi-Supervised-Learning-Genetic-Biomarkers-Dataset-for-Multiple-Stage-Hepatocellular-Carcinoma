import pandas as pd

# Read the list of matched indices
matched_indices_df = pd.read_csv('matched_indices.csv')

# Read the TCGA dataset
tcga_df = pd.read_csv('TCGA_Common.csv')

# Create an empty DataFrame to store the extracted rows
result_df = pd.DataFrame()

# Extract rows based on matched indices
for index in matched_indices_df['Matched_Indices']:
    result_df = pd.concat([result_df, tcga_df.iloc[[index-2]]])

# Append the extracted rows to Lubeck2.csv
result_df.to_csv('Lubeck4.csv', mode='a', header=False, index=False)