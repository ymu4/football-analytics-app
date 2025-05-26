import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from datetime import timedelta
import joblib
from sklearn.preprocessing import StandardScaler

def get_enhanced_historical_stats(df, team, date, n_matches=5):
    """Enhanced version of historical stats calculation"""
    team_matches = df[
         ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)

    if len(team_matches) == 0:
        return {
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'form_points': 0,
            'max_goals_scored': 0,
            'scoring_consistency': 0,
            'defensive_consistency': 0,
            'home_scoring_rate': 0,
            'away_scoring_rate': 0,
            'win_streak': 0,
            'momentum_factor': 0,
            'clean_sheet_rate': 0,
            'days_since_last_match': 30
        }

    goals_scored = []
    goals_conceded = []
    points = []
    home_goals = []
    away_goals = []
    results = []
    dates = []

    for _, match in team_matches.iterrows():
        dates.append(match['Date'])
        if match['HomeTeam'] == team:
            goals_scored.append(match['FTHG'])
            goals_conceded.append(match['FTAG'])
            home_goals.append(match['FTHG'])
            if match['FTR'] == 'H':
                points.append(3)
                results.append(1)
            elif match['FTR'] == 'D':
                points.append(1)
                results.append(0.5)
            else:
                points.append(0)
                results.append(0)
        else:
            goals_scored.append(match['FTAG'])
            goals_conceded.append(match['FTHG'])
            away_goals.append(match['FTAG'])
            if match['FTR'] == 'A':
                points.append(3)
                results.append(1)
            elif match['FTR'] == 'D':
                points.append(1)
                results.append(0.5)
            else:
                points.append(0)
                results.append(0)

    days_since_last = (date - dates[0]).days if dates else 30

    return {
        'avg_goals_scored': np.mean(goals_scored),
        'avg_goals_conceded': np.mean(goals_conceded),
        'form_points': np.mean(points),
        'max_goals_scored': np.max(goals_scored),
        'scoring_consistency': np.std(goals_scored),
        'defensive_consistency': np.std(goals_conceded),
        'home_scoring_rate': np.mean(home_goals) if home_goals else 0,
        'away_scoring_rate': np.mean(away_goals) if away_goals else 0,
        'win_streak': sum(1 for r in results if r == 1),
        'momentum_factor': sum(r * (i+1) for i, r in enumerate(reversed(results))),
        'clean_sheet_rate': sum(1 for g in goals_conceded if g == 0) / len(goals_conceded),
        'days_since_last_match': days_since_last
    }

def get_enhanced_head_to_head(df, home_team, away_team, date, n_matches=3):
    """Enhanced head to head analysis"""
    h2h_matches = df[
        (
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
        ) &
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)

    if len(h2h_matches) == 0:
        return {
            'home_win_rate': 0,
            'away_win_rate': 0,
            'home_dominance': 0
        }

    home_wins = 0
    away_wins = 0

    for _, match in h2h_matches.iterrows():
        if match['HomeTeam'] == home_team:
            if match['FTR'] == 'H':
                home_wins += 1
            elif match['FTR'] == 'A':
                away_wins += 1
        else:
            if match['FTR'] == 'H':
                away_wins += 1
            elif match['FTR'] == 'A':
                home_wins += 1

    return {
        'home_win_rate': home_wins / len(h2h_matches),
        'away_win_rate': away_wins / len(h2h_matches),
        'home_dominance': (home_wins - away_wins) / len(h2h_matches)
    }

def prepare_enhanced_features(df):
    """Prepare enhanced feature set with additional metrics"""
    print("Starting enhanced feature preparation...")
    features = []
    df = df.sort_values('Date')

    for idx, match in df.iterrows():
        try:
            home_stats = get_enhanced_historical_stats(df, match['HomeTeam'], match['Date'])
            away_stats = get_enhanced_historical_stats(df, match['AwayTeam'], match['Date'])
            h2h_stats = get_enhanced_head_to_head(df, match['HomeTeam'], match['AwayTeam'], match['Date'])

            # Calculate betting odds ratios
            total_odds = match['B365H'] + match['B365A'] + match['B365D']
            home_odds_ratio = match['B365H'] / total_odds if total_odds > 0 else 0
            away_odds_ratio = match['B365A'] / total_odds if total_odds > 0 else 0
            draw_odds_ratio = match['B365D'] / total_odds if total_odds > 0 else 0

            features.append({
                # Basic stats
                'home_avg_goals_scored': home_stats['avg_goals_scored'],
                'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
                'home_form_points': home_stats['form_points'],
                'away_avg_goals_scored': away_stats['avg_goals_scored'],
                'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
                'away_form_points': away_stats['form_points'],

                # Enhanced team metrics
                'home_max_goals': home_stats['max_goals_scored'],
                'home_scoring_consistency': home_stats['scoring_consistency'],
                'home_defensive_consistency': home_stats['defensive_consistency'],
                'home_win_streak': home_stats['win_streak'],
                'home_momentum': home_stats['momentum_factor'],
                'home_clean_sheet_rate': home_stats['clean_sheet_rate'],
                'home_days_rest': home_stats['days_since_last_match'],

                'away_max_goals': away_stats['max_goals_scored'],
                'away_scoring_consistency': away_stats['scoring_consistency'],
                'away_defensive_consistency': away_stats['defensive_consistency'],
                'away_win_streak': away_stats['win_streak'],
                'away_momentum': away_stats['momentum_factor'],
                'away_clean_sheet_rate': away_stats['clean_sheet_rate'],
                'away_days_rest': away_stats['days_since_last_match'],

                # Head-to-head metrics
                'h2h_home_win_rate': h2h_stats['home_win_rate'],
                'h2h_away_win_rate': h2h_stats['away_win_rate'],
                'h2h_home_dominance': h2h_stats['home_dominance'],

                # Betting odds ratios
                'home_odds_ratio': home_odds_ratio,
                'away_odds_ratio': away_odds_ratio,
                'draw_odds_ratio': draw_odds_ratio,

                # Advanced combined metrics
                'momentum_difference': home_stats['momentum_factor'] - away_stats['momentum_factor'],
                'form_difference': home_stats['form_points'] - away_stats['form_points'],

                # Match information
                'Date': match['Date'],
                'HomeTeam': match['HomeTeam'],
                'AwayTeam': match['AwayTeam'],
                'FTR': match['FTR']
            })

        except Exception as e:
            print(f"Error processing match {idx}: {e}")
            continue

    return pd.DataFrame(features)

def train_enhanced_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train and evaluate enhanced models"""
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=4,
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.85,
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=2,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=42
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Training predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)

        # Test predictions
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        results[name] = {
            'model': model,
            'train_pred': y_train_pred,
            'train_proba': y_train_proba,
            'test_pred': y_test_pred,
            'test_proba': y_test_proba
        }

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n{name} Top 10 Feature Importances:")
            print(importances.head(10))
        elif hasattr(model, 'coef_'):
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': abs(model.coef_[0])
            }).sort_values('importance', ascending=False)

            print(f"\n{name} Top 10 Feature Importances:")
            print(importances.head(10))

    return results

def save_trained_model(model, feature_cols, df_features):
    """Save the trained model and preprocessing components"""
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(df_features[feature_cols])
    
    # Save the model, feature columns, and scaler
    model_components = {
        'model': model,
        'feature_cols': feature_cols,
        'scaler': scaler
    }
    
    joblib.dump(model_components, 'football_prediction_model.joblib')
    print("Model and components saved successfully!")



def main():
    print("Starting prediction system...")

    # Read and combine data
    try:
        raw_data_1 = pd.read_csv('2019-2020 Fixtures.csv', parse_dates=['Date'], dayfirst=True)
        raw_data_2 = pd.read_csv('2020-2021 Fixtures.csv', parse_dates=['Date'], dayfirst=True)
        df = pd.concat([raw_data_1, raw_data_2], ignore_index=True)
        df = df.sort_values('Date')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare enhanced features
    df_features = prepare_enhanced_features(df)

    # Define prediction target
    df_features['home_win'] = (df_features['FTR'] == 'H').astype(int)

    # Select features for modeling (excluding betting odds)
    feature_cols = [col for col in df_features.columns if col not in [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'home_win'
    ]]

    # Prepare features
    X = scale(df_features[feature_cols])
    y_home_win = df_features['home_win']

    # Split data temporally (80-20 split)
    train_date = df_features['Date'].quantile(0.8)
    train_mask = df_features['Date'] <= train_date

    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_home_win_train = y_home_win[train_mask]
    y_home_win_test = y_home_win[~train_mask]

    # Train models for home win prediction
    print("\nTraining models for home win prediction...")
    home_win_results = train_enhanced_models(
        X_train, X_test,
        y_home_win_train, y_home_win_test,
        feature_cols
    )

    # Print performance metrics for home win prediction
    print("\n=== Home Win Prediction Results ===")
    for model_name, results in home_win_results.items():
        print(f"\n{model_name} Results:")
        print("\nTraining Set Performance:")
        print(classification_report(y_home_win_train, results['train_pred']))
        print("\nTest Set Performance:")
        print(classification_report(y_home_win_test, results['test_pred']))

        # Confusion matrices
        print("\nTraining Confusion Matrix:")
        print(confusion_matrix(y_home_win_train, results['train_pred']))
        print("\nTest Confusion Matrix:")
        print(confusion_matrix(y_home_win_test, results['test_pred']))

    rf_model = home_win_results['Random Forest']['model']
    save_trained_model(rf_model, feature_cols, df_features)



    # After training, save the Random Forest model (best performer)
    rf_model = home_win_results['Random Forest']['model']
    save_trained_model(rf_model, feature_cols, df_features)
if __name__ == "__main__":
    main()
