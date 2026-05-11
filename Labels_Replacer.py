import pandas as pd

# Read the list of matched indices
matched_indices_df = pd.read_csv('matched_indices.csv')

# Read the Lubeck2.csv file
lubeck_df = pd.read_csv('Lubeck4.csv')

# Read the semi_supervised_results10.csv file
semi_supervised_df = pd.read_csv('predictions3.csv')

# Extract rows based on matched indices and replace true_label with predicted_label
for index, row in matched_indices_df.iterrows():
    lubeck_index = index + 670  # Adjust for starting from row 30 in Lubeck2.csv (originally index + 28, 201, 411)
    true_label = lubeck_df.at[lubeck_index, 'symbol']
    predicted_label = semi_supervised_df.at[row['Matched_Indices']-2, 'Predicted_Labels']

    # Replace true_label with predicted_label in Lubeck2.csv
    lubeck_df.at[lubeck_index, 'symbol'] = predicted_label
    print(predicted_label)
# Save the updated Lubeck2.csv file
lubeck_df.to_csv('Lubeck4.csv', index=False)
