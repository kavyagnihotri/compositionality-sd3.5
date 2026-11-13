import pandas as pd

# Load the CSV files into DataFrames
df1 = pd.read_csv('clic_weight_comparison.csv')
df2 = pd.read_csv('cogvlm_weight_comparison.csv')

# Merge the DataFrames on the 'Layer Name' column.
# The 'suffixes' parameter automatically renames the overlapping 'Frobenius Norm' columns.
merged_df = pd.merge(df1, df2, on='Layer Name', suffixes=('_clic', '_cogvlm'))

# Save the merged data to a new CSV file
merged_df.to_csv('merged_output.csv', index=False)

print("Files successfully merged into 'merged_output.csv'!")