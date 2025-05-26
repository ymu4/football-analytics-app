import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import classification_report

def train_and_evaluate_model():
    """Train and save a simplified model using only the most important features"""
    
    # Load both training and testing sets
    train_df = pd.read_excel('training_set_processed.xlsx')
    test_df = pd.read_excel('testing_set_processed.xlsx')
    
    # Select the most important features based on the analysis
    important_features = [
        'Def Pen_Possession',  # Defensive actions in penalty area
        'defensive_contribution',  # Combined defensive metrics
        'Att Pen_Possession',  # Attacking penalty area entries
        'SoT/90_Shooting',  # Shots on target per 90
        'Tkl_Defensive',  # Tackles
        'Int_Defensive',  # Interceptions
        'Touches_Possession',  # Ball touches
        'Carries_Possession',  # Ball carries
        'Blocks_Defensive',  # Blocks
        'Gls_Shooting', # Goals
        'Clr_Defensive'
        # 'Def 3rd_Possession',
        # 'Sh_Shooting'
        # 'Fls_Passing'


    ]
    
    # Prepare features and target for both sets
    X_train = train_df[important_features]
    y_train = train_df['Position']
    
    X_test = test_df[important_features]
    y_test = test_df['Position']
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Save the model and feature names
    model_data = {
        'pipeline': pipeline,
        'feature_names': important_features,
        'feature_descriptions': {
            'Def Pen_Possession': 'Defensive Actions in Penalty Area',
            'defensive_contribution': 'Overall Defensive Contribution',
            'Att Pen_Possession': 'Attacking Penalty Area Entries',
            'SoT/90_Shooting': 'Shots on Target per 90 mins',
            'Tkl_Defensive': 'Tackles Made',
            'Int_Defensive': 'Interceptions Made',
            'Touches_Possession': 'Total Ball Touches',
            'Carries_Possession': 'Ball Carries',
            'Blocks_Defensive': 'Blocks Made',
            'Gls_Shooting': 'Goals Scored',
            'Clr_Defensive': 'Clearance Defensive'
        }
    }
    
    joblib.dump(model_data, 'player_position_model.joblib')
    
    # Print model performance
    print("\nModel Performance Evaluation:")
    print("\nTraining Set Performance:")
    train_preds = pipeline.predict(X_train)
    print(classification_report(y_train, train_preds))
    
    print("\nTest Set Performance:")
    test_preds = pipeline.predict(X_test)
    print(classification_report(y_test, test_preds))

if __name__ == "__main__":
    train_and_evaluate_model()