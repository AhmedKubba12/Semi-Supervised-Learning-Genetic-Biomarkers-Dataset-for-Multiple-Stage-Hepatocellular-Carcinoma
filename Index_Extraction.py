import pandas as pd

# Read the CSV file
df = pd.read_csv('predictions_of_Lubeck2.csv')
print(df)
# Create an empty list to store the indices that meet the conditions
matched_indices = []

# Loop through the rows of the DataFrame
for index, row in df.iterrows():
    true_label = row['True_Labels']
    predicted_label = row['Predicted_Labels']

    # Check conditions for HCC and eHCC or pHCC
    if true_label == 'HCC' and (predicted_label == 'ehcc' or predicted_label == 'phcc'):
        matched_indices.append(index+2)

    # Check conditions for Transition and LGDN or HGDN or SL
    elif true_label == 'Transation' and (predicted_label == 'lgdn' or predicted_label == 'hgdn'):
        matched_indices.append(index+2)
    
    # Check conditions for Normal and SL
    elif true_label == 'Normal' and (predicted_label == 'sl'):
        matched_indices.append(index+2)

# Create a new DataFrame with the matched indices
result_df = pd.DataFrame({'Matched_Indices': matched_indices})

# Save the result DataFrame to a new CSV file
result_df.to_csv('matched_indices2.csv', index=False)
