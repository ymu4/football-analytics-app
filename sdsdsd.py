import pandas as pd

# Define the important features
important_features = [
    'Def Pen_Possession',
    'defensive_contribution',
    'Att Pen_Possession',
    'SoT/90_Shooting',
    'Tkl_Defensive',
    'Int_Defensive',
    'Touches_Possession',
    'Carries_Possession',
    'Blocks_Defensive',
    'Gls_Shooting',
    'Position'
]

# Read the Excel file
df = pd.read_excel('live_testing_set_processed.xlsx')

# Add 'Player' to the list of columns to keep
columns_to_keep = ['Player', 'Position'] + important_features
# columns_to_keep = ['Position'] + important_features

# Create new dataframe with only the selected columns
df_filtered = df[columns_to_keep]

# Save the filtered dataframe to a new Excel file
output_filename = 'live_testing_set_filtered.xlsx'
df_filtered.to_excel(output_filename, index=False)

print(f"File saved successfully as {output_filename}")