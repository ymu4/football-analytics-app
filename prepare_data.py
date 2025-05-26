# import pandas as pd
# import numpy as np

# def process_excel_file(excel_file_path, output_file_path):
#     """
#     Process the Excel file to keep only specific columns that match Challenge_2_data.csv
#     plus the 'Pos' column from each sheet.
    
#     Args:
#         excel_file_path (str): Path to the input Excel file
#         output_file_path (str): Path to save the processed Excel file
#     """
#     # Read all sheets from the Excel file
#     excel_file = pd.ExcelFile(excel_file_path)
    
#     # Define the columns from Challenge_2_data.csv
#     challenge_2_columns = [
#         'Rk', 'Nation', 'Squad', 'Age', 'Born', '90s', 'Gls', 'Sh', 'SoT', 'SoT%',
#         'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Dist', 'FK', 'PK', 'PKatt', 'xG',
#         'npxG', 'npxG/Sh', 'G-xG', 'np:G-xG', 'Cmp', 'Att', 'Cmp%', 'TotDist',
#         'PrgDist', 'Cmp.1', 'Att.1', 'Cmp%.1', 'Cmp.2', 'Att.2', 'Cmp%.2', 'Cmp.3',
#         'Att.3', 'Cmp%.3', 'Ast', 'xAG', 'xA', 'A-xAG', 'KP', 'FianlThirdPass',
#         'PPA', 'CrsPA', 'PrgP', 'Att.4', 'Live', 'Dead', 'FK.1', 'TB', 'Sw', 'Crs',
#         'TI', 'CK', 'In', 'Out', 'Str', 'Cmp.4', 'Off', 'Blocks', 'SCA', 'SCA90',
#         'PassLive', 'PassDead', 'TO', 'Sh.1', 'Fld', 'Def', 'GCA', 'GCA90',
#         'PassLive.1', 'PassDead.1', 'TO.1', 'Sh.2', 'Fld.1', 'Def.1', 'Tkl',
#         'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Tkl.1', 'Att.5', 'Tkl%', 'Lost',
#         'Blocks.1', 'Sh.3', 'Pass', 'Int', 'Tkl+Int', 'Clr', 'Err', 'Touches',
#         'Def Pen', 'Def 3rd.1', 'Mid 3rd.1', 'Att 3rd.1', 'Att Pen', 'Live.1',
#         'Att.6', 'Succ', 'Succ%', 'Tkld', 'Tkld%', 'Carries', 'TotDist.1',
#         'PrgDist.1', 'PrgC', 'FinalThirdPos', 'CPA', 'Mis', 'Dis', 'Rec', 'PrgR',
#         'CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld.2', 'Off.1', 'Crs.1', 'Int.1',
#         'TklW.1', 'PKwon', 'PKcon', 'OG', 'Recov', 'Won', 'Lost.1', 'Won%', 'GA',
#         'PKA', 'FK.2', 'CK.1', 'OG.1', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90',
#         'Cmp.5', 'Att.7', 'Cmp%.4', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen',
#         'Att.8', 'Launch%.1', 'AvgLen.1', 'Opp', 'Stp', 'Stp%', '#OPA',
#         '#OPA/90', 'AvgDist'
#     ]

#     # Create a dictionary to store processed dataframes
#     processed_sheets = {}

#     # Process each sheet
#     for sheet_name in excel_file.sheet_names:
#         # Read the sheet
#         df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
#         # Get the columns that exist in both the sheet and challenge_2_columns
#         existing_columns = [col for col in challenge_2_columns if col in df.columns]
        
#         # Add 'Pos' column if it exists
#         if 'Pos' in df.columns:
#             existing_columns = ['Pos'] + existing_columns
            
#         # Keep only the specified columns
#         if existing_columns:
#             processed_sheets[sheet_name] = df[existing_columns]

#     # Save to a new Excel file
#     with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
#         for sheet_name, df in processed_sheets.items():
#             df.to_excel(writer, sheet_name=sheet_name, index=False)

#     return processed_sheets

# # Usage example:
# if __name__ == "__main__":
#     input_file = "Premier League Players 23_24 Stats.xlsx"
#     output_file = "Processed_Premier_League_Stats.xlsx"
#     processed_data = process_excel_file(input_file, output_file)


# import pandas as pd

# def clean_excel_file(input_file_path, output_file_path):
#     """
#     Clean the Excel file by:
#     1. Removing duplicate headers (rows where Pos = 'Pos')
#     2. Removing duplicate columns
#     3. Cleaning up the data structure
    
#     Args:
#         input_file_path (str): Path to the input Excel file
#         output_file_path (str): Path to save the cleaned Excel file
#     """
#     # Read all sheets from the Excel file
#     excel_file = pd.ExcelFile(input_file_path)
    
#     # Create a dictionary to store cleaned dataframes
#     cleaned_sheets = {}

#     # Process each sheet
#     for sheet_name in excel_file.sheet_names:
#         # Read the sheet
#         df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
#         # Remove rows where Pos = 'Pos' (duplicate headers)
#         df = df[df['Pos'] != 'Pos']
        
#         # Reset the index after removing rows
#         df = df.reset_index(drop=True)
        
#         # Get list of duplicate columns
#         duplicate_cols = df.columns[df.columns.duplicated()]
        
#         # Keep only the first instance of duplicate columns
#         if len(duplicate_cols) > 0:
#             df = df.loc[:, ~df.columns.duplicated()]
        
#         # Store the cleaned dataframe
#         cleaned_sheets[sheet_name] = df

#     # Save to a new Excel file
#     with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
#         for sheet_name, df in cleaned_sheets.items():
#             df.to_excel(writer, sheet_name=sheet_name, index=False)

#     return cleaned_sheets

# # Usage example:
# if __name__ == "__main__":
#     input_file = "Processed_Premier_League_Stats.xlsx"
#     output_file = "Cleaned_Premier_League_Stats.xlsx"
#     cleaned_data = clean_excel_file(input_file, output_file)


import pandas as pd
import numpy as np

def merge_excel_sheets(input_file_path, output_file_path):
    """
    Merge all sheets into one dataframe, handling goalkeeper stats appropriately
    and ordering columns according to Challenge_2_data.csv
    """
    # Read all sheets
    excel_file = pd.ExcelFile(input_file_path)
    
    # Initialize dictionary to store dataframes
    dfs = {}
    
    # First, create a player identifier combining multiple fields to ensure unique matching
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        # Create a unique identifier for each player including Rk
        df['player_id'] = df.apply(lambda x: f"{x['Squad']}_{x['Pos']}_{x['Nation']}_{x['Age']}_{x['Born']}_{x['Rk']}", axis=1)
        if sheet_name != 'Goalkeeper Stats':
            dfs[sheet_name] = df
    
    # Start with the shooting stats as base
    base_df = dfs['Shooting Stats'].copy()
    
    # Merge other sheets one by one using the player_id
    for sheet_name, df in dfs.items():
        if sheet_name != 'Shooting Stats':
            # Remove common columns before merging
            cols_to_use = [col for col in df.columns if col not in ['Pos', 'Nation', 'Squad', 'Age', 'Born', '90s', 'Rk']]
            merge_df = df[['player_id'] + cols_to_use]
            base_df = pd.merge(base_df, merge_df, on='player_id', how='outer')
    
    # Handle goalkeeper data
    gk_df = pd.read_excel(excel_file, sheet_name='Goalkeeper Stats')
    gk_df['player_id'] = gk_df.apply(lambda x: f"{x['Squad']}_{x['Pos']}_{x['Nation']}_{x['Age']}_{x['Born']}_{x['Rk']}", axis=1)
    
    # Create template for goalkeeper data
    gk_template = pd.DataFrame(0, index=gk_df.index, columns=base_df.columns)
    
    # Fill in the goalkeeper information
    common_cols = ['Pos', 'Nation', 'Squad', 'Age', 'Born', '90s', 'Rk', 'player_id']
    for col in common_cols:
        if col in gk_df.columns:
            gk_template[col] = gk_df[col]
    
    # Add goalkeeper-specific columns to base_df and fill with 0s
    gk_specific_cols = ['GA', 'PKA', 'FK.2', 'CK.1', 'OG.1', 'PSxG', 'PSxG/SoT', 
                       'PSxG+/-', '/90', 'Cmp.5', 'Att.7', 'Cmp%.4', 'Att (GK)', 
                       'Thr', 'Launch%', 'AvgLen', 'Launch%.1', 'AvgLen.1', 'Opp', 
                       'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']
    
    for col in gk_specific_cols:
        if col in gk_df.columns:
            if col not in base_df.columns:
                base_df[col] = 0
            gk_template[col] = gk_df[col]
    
    # Combine outfield players and goalkeepers
    final_df = pd.concat([base_df, gk_template], ignore_index=True)
    
    # Fill NaN values with 0
    final_df = final_df.fillna(0)
    
    # Remove the temporary player_id column
    final_df = final_df.drop('player_id', axis=1)
    
    # Order columns according to Challenge_2_data.csv
    desired_column_order = [
        'Rk', 'Nation', 'Squad', 'Age', 'Born', '90s', 'Gls', 'Sh', 'SoT', 'SoT%',
        'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Dist', 'FK', 'PK', 'PKatt', 'xG',
        'npxG', 'npxG/Sh', 'G-xG', 'np:G-xG', 'Cmp', 'Att', 'Cmp%', 'TotDist',
        'PrgDist', 'Cmp.1', 'Att.1', 'Cmp%.1', 'Cmp.2', 'Att.2', 'Cmp%.2', 'Cmp.3',
        'Att.3', 'Cmp%.3', 'Ast', 'xAG', 'xA', 'A-xAG', 'KP', 'FianlThirdPass',
        'PPA', 'CrsPA', 'PrgP', 'Att.4', 'Live', 'Dead', 'FK.1', 'TB', 'Sw', 'Crs',
        'TI', 'CK', 'In', 'Out', 'Str', 'Cmp.4', 'Off', 'Blocks', 'SCA', 'SCA90',
        'PassLive', 'PassDead', 'TO', 'Sh.1', 'Fld', 'Def', 'GCA', 'GCA90',
        'PassLive.1', 'PassDead.1', 'TO.1', 'Sh.2', 'Fld.1', 'Def.1', 'Tkl',
        'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Tkl.1', 'Att.5', 'Tkl%', 'Lost',
        'Blocks.1', 'Sh.3', 'Pass', 'Int', 'Tkl+Int', 'Clr', 'Err', 'Touches',
        'Def Pen', 'Def 3rd.1', 'Mid 3rd.1', 'Att 3rd.1', 'Att Pen', 'Live.1',
        'Att.6', 'Succ', 'Succ%', 'Tkld', 'Tkld%', 'Carries', 'TotDist.1',
        'PrgDist.1', 'PrgC', 'FinalThirdPos', 'CPA', 'Mis', 'Dis', 'Rec', 'PrgR',
        'CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld.2', 'Off.1', 'Crs.1', 'Int.1',
        'TklW.1', 'PKwon', 'PKcon', 'OG', 'Recov', 'Won', 'Lost.1', 'Won%', 'GA',
        'PKA', 'FK.2', 'CK.1', 'OG.1', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90',
        'Cmp.5', 'Att.7', 'Cmp%.4', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen',
        'Launch%.1', 'AvgLen.1', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist'
    ]
    
    # Add Pos column at the beginning and only include columns that exist
    existing_cols = ['Pos'] + [col for col in desired_column_order if col in final_df.columns]
    final_df = final_df[existing_cols]
    
    # Save to Excel
    final_df.to_excel(output_file_path, index=False)
    
    return final_df

# Usage example:
if __name__ == "__main__":
    input_file = "Cleaned_Premier_League_Stats.xlsx"
    output_file = "Merged_Premier_League_Stats.xlsx"
    merged_data = merge_excel_sheets(input_file, output_file)