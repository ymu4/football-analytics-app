import pandas as pd

def filter_columns(input_file, output_file):
    # List of columns to keep
    columns_to_keep = [
        'Pos',
        'Def Pen_Possession', 
        'Def 3rd_Possession', 
        'Mid 3rd_Possession',
        'Att 3rd_Possession', 
        'TI_PassTypes', 
        'Clr_Defensive',
        'Att.3_Passing', 
        'Att Pen_Possession', 
        'PrgR_Possession'
    ]
    
    try:
        # Read the Excel file
        df = pd.read_excel(input_file)
        
        # Select only the specified columns
        # Using reindex to maintain the order of columns as specified
        filtered_df = df.reindex(columns=columns_to_keep)
        
        # Save the filtered DataFrame to a new Excel file
        filtered_df.to_excel(output_file, index=False)
        print(f"Successfully created {output_file} with selected columns")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage
input_file = './processed_data/merged_test_data.xlsx'
output_file = 'filtered_data.xlsx'
filter_columns(input_file, output_file)