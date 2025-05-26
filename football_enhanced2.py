import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import joblib
import os

def load_and_merge_data(file_path, save_path=None):
    """
    Load and merge all sheets from Premier League stats Excel file
    Args:
        file_path: Path to the Excel file
        save_path: Optional path to save the merged dataset
    """
    sheets = pd.read_excel(file_path, sheet_name=None, na_values=['', ' ', 'NA', 'N/A'])
    processed_sheets = []
    
    for sheet_name, df in sheets.items():
        print(f"\nProcessing sheet: {sheet_name}")
        print(f"Initial shape: {df.shape}")
        sheet_df = df.copy()
        
        # Basic cleaning
        sheet_df['Player'] = sheet_df['Player'].str.strip()
        sheet_df['Squad'] = sheet_df['Squad'].str.strip()
        sheet_df = sheet_df[~(sheet_df['Player'].isin(['Player', '']) |
                            sheet_df['Squad'].isin(['Squad', '']))]
        
        # Handle 90s column
        if '90s' in sheet_df.columns:
            sheet_df.rename(columns={'90s': 'Minutes_90s'}, inplace=True)
            sheet_df['Minutes_90s'] = pd.to_numeric(sheet_df['Minutes_90s'], errors='coerce')
        
        # Create unique identifier and handle column naming
        sheet_df['Player_ID'] = sheet_df.apply(
            lambda x: f"{x['Player']}_{x['Squad']}_{x['Rk']}", axis=1)
        prefix = sheet_name.split(' ')[0]
        base_cols = ['Player', 'Squad', 'Nation', 'Pos', 'Rk', 'Player_ID', 'Minutes_90s']
        cols_to_rename = [col for col in sheet_df.columns if col not in base_cols]
        sheet_df.rename(columns={col: f"{col}_{prefix}" for col in cols_to_rename},
                       inplace=True)
        
        print(f"Processed shape: {sheet_df.shape}")
        processed_sheets.append(sheet_df)
    
    # Merge all sheets
    final_df = processed_sheets[0]
    for i, right_df in enumerate(processed_sheets[1:], 1):
        right_cols = [col for col in right_df.columns if col not in final_df.columns]
        right_cols.append('Player_ID')
        right_df_slim = right_df[right_cols]
        print(f"\nMerging sheet {i+1}")
        print(f"Current shape: {final_df.shape}")
        final_df = pd.merge(
            final_df,
            right_df_slim,
            on='Player_ID',
            how='outer',
            validate='1:1'
        )
        print(f"After merge: {final_df.shape}")
    
    final_df = final_df.drop('Player_ID', axis=1)
    
    # Save merged dataset if save_path is provided
    if save_path:
        print(f"\nSaving merged dataset to: {save_path}")
        if save_path.endswith('.csv'):
            final_df.to_csv(save_path, index=False)
        elif save_path.endswith('.xlsx'):
            final_df.to_excel(save_path, index=False)
        print("Dataset saved successfully!")
    
    return final_df

def process_and_train_model(train_file_path, test_file_path):
    """
    Process data and train the model with selected features using GridSearchCV
    """
    # Create output directory if it doesn't exist
    output_dir = 'processed_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save paths for merged datasets
    train_merged_path = os.path.join(output_dir, 'merged_train_data.xlsx')
    test_merged_path = os.path.join(output_dir, 'merged_test_data.xlsx')
    
    # Important features to select
    important_features = [
        'Def Pen_Possession',
        'Def 3rd_Possession',
        'Mid 3rd_Possession',
        'Att 3rd_Possession',
        'TI_PassTypes',
        'Clr_Defensive',
        'Att.3_Passing',
        'Att Pen_Possession',
        'PrgR_Possession',

        #  'PrgDist_Passing',
        # 'Sh_Shooting',
        # 'Lost_Passing',
        # 'Dead_PassTypes',
        # # 'Tkl+Int_Defensive',
        # 'PrgP_Passing'
    ]
    
    # Feature descriptions for documentation
    feature_descriptions = {
        'Def Pen_Possession': 'Defensive Actions in Penalty Area',
        'Def 3rd_Possession': 'Actions in Defensive Third',
        'Mid 3rd_Possession': 'Actions in Middle Third',
        'Att 3rd_Possession': 'Actions in Attacking Third',
        'TI_PassTypes': 'Total Pass Types',
        'Clr_Defensive': 'Clearances Made',
        'Att.3_Passing': 'Passes into Final Third',
        'Att Pen_Possession': 'Actions in Attacking Penalty Area',
        'PrgR_Possession': 'Progressive Carries',
        # 'PrgDist_Passing': 'Progressive Distance in Passing',
        # 'Sh_Shooting': 'Total Shots',
        # 'Lost_Passing': 'Passes Lost',
        # 'Dead_PassTypes': 'Dead Ball Passes',
        # 'Tkl+Int_Defensive': 'Combined Tackles and Interceptions',
        # 'PrgP_Passing': 'Progressive Passes'
    }
    
    print("Loading and processing training data...")
    train_df = load_and_merge_data(train_file_path, train_merged_path)
    print("Loading and processing test data...")
    test_df = load_and_merge_data(test_file_path, test_merged_path)
    
    # Clean positions and remove rows with NaN positions
    for i, df in enumerate([train_df, test_df]):
        df['Position'] = df['Pos'].apply(lambda x: str(x).split(',')[0].strip())
        # Remove rows where Position is not in our target classes
        valid_mask = df['Position'].isin(['DF', 'MF', 'FW', 'GK'])
        df_cleaned = df[valid_mask].copy()
        
        if i == 0:
            train_df = df_cleaned
        else:
            test_df = df_cleaned
            
    print("\nAfter cleaning positions:")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Prepare features and target
    X_train = train_df[important_features]
    y_train = train_df['Position']
    X_test = test_df[important_features]
    y_test = test_df['Position']
    
    # Check for missing values
    print("\nMissing values in training set:")
    print(X_train.isnull().sum())
    
    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("\nClass distribution in training set:")
    print(y_train.value_counts())
    
    # Define the pipeline with imputation
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [ 150], 
        'classifier__max_depth': [ None],
        'classifier__min_samples_split': [4],
        'classifier__min_samples_leaf': [2],
        'classifier__max_features': ['sqrt'],
        'classifier__max_samples': [0.9],
        'classifier__criterion': ['gini'],
        'classifier__bootstrap': [True],
        'classifier__class_weight': ['balanced_subsample'],
        'imputer__strategy': ['mean']
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create GridSearchCV object
    print("\nPerforming grid search...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='balanced_accuracy',
        verbose=2
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Save model and metadata
    model_data = {
        'pipeline': best_model,
        'feature_names': important_features,
        'feature_descriptions': feature_descriptions,
        'best_params': grid_search.best_params_,
        'cv_results': grid_search.cv_results_
    }
    
    print("\nSaving model and features...")
    joblib.dump(model_data, os.path.join(output_dir, 'player_position_model.joblib'))
    
    # Evaluate model performance
    print("\nModel Performance Evaluation:")
    print("\nTraining Set Performance:")
    train_preds = best_model.predict(X_train)
    print(classification_report(y_train, train_preds))
    
    print("\nTest Set Performance:")
    test_preds = best_model.predict(X_test)
    print(classification_report(y_test, test_preds))
    
    # Print cross-validation scores
    print("\nCross-validation scores:")
    cv_scores = grid_search.cv_results_['mean_test_score']
    cv_std = grid_search.cv_results_['std_test_score']
    print(f"Mean CV score: {grid_search.best_score_:.4f} (+/- {cv_std[grid_search.best_index_]:.4f})")
    
    # Print feature importances
    print("\nTop 15 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': important_features,
        'importance': best_model.named_steps['classifier'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nFeature Importance Rankings:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
        
    # Add feature importances to model data
    model_data['feature_importances'] = feature_importance.to_dict('records')
    
    return model_data

if __name__ == "__main__":
    train_file = 'Premier League Players 23_24 Stats_train.xlsx'
    test_file = 'Premier League Players 23_24 Stats_test.xlsx'
    model_data = process_and_train_model(train_file, test_file)