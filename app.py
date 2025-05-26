import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px



# Load the saved model and components
@st.cache_resource
def load_model():
    model_components = joblib.load('football_prediction_model.joblib')
    return model_components

def calculate_features(home_team_data, away_team_data, betting_odds):
    """Calculate features from input data"""
    
    # Calculate home team stats
    home_goals_scored = [match['goals_scored'] for match in home_team_data]
    home_goals_conceded = [match['goals_conceded'] for match in home_team_data]
    home_results = [match['result'] for match in home_team_data]
    
    # Calculate away team stats
    away_goals_scored = [match['goals_scored'] for match in away_team_data]
    away_goals_conceded = [match['goals_conceded'] for match in away_team_data]
    away_results = [match['result'] for match in away_team_data]
    
    # Calculate head-to-head stats (simplified since we don't have historical h2h data)
    h2h_stats = {
        'h2h_home_win_rate': 0.5,  # Default to neutral
        'h2h_away_win_rate': 0.5,  # Default to neutral
        'h2h_home_dominance': 0.0   # Default to neutral
    }
    
    # Calculate days rest (simplified since we don't have match dates)
    rest_stats = {
        'home_days_rest': 7,  # Default to one week
        'away_days_rest': 7   # Default to one week
    }
    
    # Convert results to points
    result_to_points = {'W': 3, 'D': 1, 'L': 0}
    home_points = [result_to_points[r] for r in home_results]
    away_points = [result_to_points[r] for r in away_results]
    
    # Calculate features
    features = {
        # Existing features
        'home_avg_goals_scored': np.mean(home_goals_scored),
        'home_avg_goals_conceded': np.mean(home_goals_conceded),
        'home_form_points': np.mean(home_points),
        'away_avg_goals_scored': np.mean(away_goals_scored),
        'away_avg_goals_conceded': np.mean(away_goals_conceded),
        'away_form_points': np.mean(away_points),
        
        # Add new features
        'home_days_rest': rest_stats['home_days_rest'],
        'away_days_rest': rest_stats['away_days_rest'],
        'h2h_home_win_rate': h2h_stats['h2h_home_win_rate'],
        'h2h_away_win_rate': h2h_stats['h2h_away_win_rate'],
        'h2h_home_dominance': h2h_stats['h2h_home_dominance'],
        
        'home_max_goals': max(home_goals_scored),
        'home_scoring_consistency': np.std(home_goals_scored),
        'home_defensive_consistency': np.std(home_goals_conceded),
        'home_win_streak': sum(1 for r in home_results if r == 'W'),
        'home_momentum': sum((i+1) * (1 if r == 'W' else 0.5 if r == 'D' else 0) 
                           for i, r in enumerate(reversed(home_results))),
        'home_clean_sheet_rate': sum(1 for g in home_goals_conceded if g == 0) / len(home_goals_conceded),
        
        'away_max_goals': max(away_goals_scored),
        'away_scoring_consistency': np.std(away_goals_scored),
        'away_defensive_consistency': np.std(away_goals_conceded),
        'away_win_streak': sum(1 for r in away_results if r == 'W'),
        'away_momentum': sum((i+1) * (1 if r == 'W' else 0.5 if r == 'D' else 0) 
                           for i, r in enumerate(reversed(away_results))),
        'away_clean_sheet_rate': sum(1 for g in away_goals_conceded if g == 0) / len(away_goals_conceded),
        
        # Betting odds ratios
        'home_odds_ratio': betting_odds['home'] / sum(betting_odds.values()),
        'away_odds_ratio': betting_odds['away'] / sum(betting_odds.values()),
        'draw_odds_ratio': betting_odds['draw'] / sum(betting_odds.values()),
        
        # Additional combined metrics
        'momentum_difference': (sum((i+1) * (1 if r == 'W' else 0.5 if r == 'D' else 0) 
                                  for i, r in enumerate(reversed(home_results))) -
                              sum((i+1) * (1 if r == 'W' else 0.5 if r == 'D' else 0) 
                                  for i, r in enumerate(reversed(away_results)))),
        'form_difference': np.mean(home_points) - np.mean(away_points)
    }
    
    return features

def predict_match_outcome(features, model_components):
    """Predict match outcome using the loaded model"""
    model = model_components['model']
    scaler = model_components['scaler']
    feature_cols = model_components['feature_cols']
    
    # Prepare features in the correct order
    feature_vector = pd.DataFrame([features])[feature_cols]
    
    # Scale features
    scaled_features = scaler.transform(feature_vector)
    
    # Get prediction probabilities
    probs = model.predict_proba(scaled_features)[0]
    
    return {
        'Home Win': probs[1],
        'Not Home Win': probs[0]  # Includes both Draw and Away Win
    }

def match_prediction_interface():
    st.header("Match Outcome Prediction Analysis")
    
    # Load model
    try:
        model_components = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Create two columns for team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Home Team")
        home_team = st.text_input("Home Team Name")
        
        st.write("Last 5 Matches")
        home_team_data = []
        
        for i in range(5):
            st.write(f"Match {i+1}")
            home_match = {}
            cols = st.columns(3)
            with cols[0]:
                home_match['goals_scored'] = st.number_input(f"Goals Scored", key=f"home_goals_{i}", min_value=0)
            with cols[1]:
                home_match['goals_conceded'] = st.number_input(f"Goals Conceded", key=f"home_conceded_{i}", min_value=0)
            with cols[2]:
                home_match['result'] = st.selectbox(f"Result", ['W', 'D', 'L'], key=f"home_result_{i}")
            home_team_data.append(home_match)

    with col2:
        st.subheader("Away Team")
        away_team = st.text_input("Away Team Name")
        
        st.write("Last 5 Matches")
        away_team_data = []
        
        for i in range(5):
            st.write(f"Match {i+1}")
            away_match = {}
            cols = st.columns(3)
            with cols[0]:
                away_match['goals_scored'] = st.number_input(f"Goals Scored", key=f"away_goals_{i}", min_value=0)
            with cols[1]:
                away_match['goals_conceded'] = st.number_input(f"Goals Conceded", key=f"away_conceded_{i}", min_value=0)
            with cols[2]:
                away_match['result'] = st.selectbox(f"Result", ['W', 'D', 'L'], key=f"away_result_{i}")
            away_team_data.append(away_match)

    # Betting odds inputs
    st.subheader("Betting Odds (Bet365)")
    odds_col1, odds_col2, odds_col3 = st.columns(3)
    with odds_col1:
        home_odds = st.number_input("Home Win Odds", min_value=1.0, step=0.01)
    with odds_col2:
        away_odds = st.number_input("Away Win Odds", min_value=1.0, step=0.01)
    with odds_col3:
        draw_odds = st.number_input("Draw Odds", min_value=1.0, step=0.01)

    # Add rest days inputs
    st.subheader("Rest Days")
    rest_col1, rest_col2 = st.columns(2)
    with rest_col1:
        home_rest = st.number_input("Home Team Days Since Last Match", min_value=1, value=7)
    with rest_col2:
        away_rest = st.number_input("Away Team Days Since Last Match", min_value=1, value=7)

    # Add head-to-head stats
    st.subheader("Historical Head-to-Head")
    h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
    with h2h_col1:
        h2h_home_wins = st.number_input("Previous H2H Home Wins (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    with h2h_col2:
        h2h_away_wins = st.number_input("Previous H2H Away Wins (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    with h2h_col3:
        h2h_dominance = st.slider("Home Team Dominance", min_value=-1.0, max_value=1.0, value=0.0, 
                                help="1.0 means home team dominated, -1.0 means away team dominated")

    betting_odds = {
        'home': home_odds,
        'away': away_odds,
        'draw': draw_odds
    }
    
    rest_stats = {
        'home_days_rest': home_rest,
        'away_days_rest': away_rest
    }
    
    h2h_stats = {
        'h2h_home_win_rate': h2h_home_wins / 100,
        'h2h_away_win_rate': h2h_away_wins / 100,
        'h2h_home_dominance': h2h_dominance
    }

    if st.button("Predict Match Outcome"):
        if not (home_team and away_team):
            st.error("Please enter both team names!")
            return
            
        # Calculate features from input data
        features = calculate_features(home_team_data, away_team_data, betting_odds)
        
        # Make prediction
        predictions = predict_match_outcome(features, model_components)
        
        # Display predictions
        st.subheader("Match Prediction")
        col1, col2, col3 = st.columns(3)
        
        # Calculate confidence score (difference between highest and lowest probability)
        home_prob = predictions['Home Win']
        not_home_prob = predictions['Not Home Win']
        confidence = abs(home_prob - not_home_prob)
        confidence_text = "High" if confidence > 0.5 else "Medium" if confidence > 0.2 else "Low"
        confidence_color = "green" if confidence > 0.5 else "orange" if confidence > 0.2 else "red"
        
        with col1:
            st.metric("Home Win Probability", f"{home_prob:.1%}")
        with col2:
            st.metric("Not Home Win Probability", f"{not_home_prob:.1%}")
        with col3:
            st.markdown(f"**Model Confidence:** ::{confidence_color}[{confidence_text}]")
            st.metric("Confidence Score", f"{confidence:.1%}")
            
        # Display interpretation
        st.subheader("Key Factors")
        factors = []
        if features['home_form_points'] > features['away_form_points']:
            factors.append(f"📈 {home_team} is in better form")
        if features['away_form_points'] > features['home_form_points']:
            factors.append(f"📈 {away_team} is in better form")
        if features['home_momentum'] > features['away_momentum']:
            factors.append(f"🔥 {home_team} has better momentum")
        if features['away_momentum'] > features['home_momentum']:
            factors.append(f"🔥 {away_team} has better momentum")
            
        for factor in factors:
            st.write(factor)


@st.cache_resource
def load_model_position():
    model_data = joblib.load('./processed_data/player_position_model.joblib')
    return model_data           


def load_and_preprocess_data():
    """Load and preprocess the dataset with proper type conversion"""
    df = pd.read_excel('./processed_data/merged_train_data.xlsx')
    
    # Remove repeated header rows
    df = df[df['Squad'] != 'Squad']
    
    # Basic cleaning
    df['Position'] = df['Pos'].apply(lambda x: str(x).split(',')[0].strip())
    df = df[df['Position'].isin(['DF', 'MF', 'FW', 'GK'])]
    
    # Convert numeric columns and handle missing values
    numeric_columns = [
        'Def Pen_Possession', 'Def 3rd_Possession', 'Mid 3rd_Possession',
        'Att 3rd_Possession', 'TI_PassTypes', 'Clr_Defensive',
        'Att.3_Passing', 'Att Pen_Possession', 'PrgR_Possession'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    return df

def create_nationality_analysis(df):
    """Create position distribution analysis"""
    st.markdown("### Player Position Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create pie chart for position distribution
        position_counts = df['Pos'].value_counts()
        fig = px.pie(
            values=position_counts.values,
            names=position_counts.index,
            title='Position Distribution in the League',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Position distribution by club
        position_by_club = pd.crosstab(df['Squad'], df['Pos'])
        position_by_club_pct = position_by_club.div(position_by_club.sum(axis=1), axis=0)
        
        fig = px.bar(
            position_by_club_pct,
            title='Position Distribution by Club',
            barmode='stack',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig)

def create_age_analysis(df):
    """Create age distribution analysis"""
    st.markdown("### Age Analysis")
    
    # Remove any NaN values for age analysis
    df_age = df.dropna(subset=['Age'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by position
        fig = px.box(
            df_age,
            x='Pos',
            y='Age',
            title='Age Distribution by Position',
            points='all',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Average age by club
        avg_age = df_age.groupby('Squad')['Age'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=avg_age.index,
            y=avg_age.values,
            title='Average Age by Club',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig)

def create_performance_analysis(df):
    """Create performance metrics visualization"""
    st.markdown("### Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top players by defensive actions
        top_defensive = df.nlargest(10, 'Def Pen_Possession')[['Player', 'Def Pen_Possession', 'Squad']]
        fig = px.bar(
            top_defensive,
            x='Player',
            y='Def Pen_Possession',
            color='Squad',
            title='Top 10 Players by Defensive Actions in Penalty Area',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Top players by progressive carries
        top_progressive = df.nlargest(10, 'PrgR_Possession')[['Player', 'PrgR_Possession', 'Squad']]
        fig = px.bar(
            top_progressive,
            x='Player',
            y='PrgR_Possession',
            color='Squad',
            title='Top 10 Players by Progressive Carries',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig)
    
    # with col2:
    #     # Shots on target visualization instead of assists
    #     top_shots = df.nlargest(10, 'SoT')[['Player', 'SoT', 'Squad']]
    #     fig = px.bar(
    #         top_shots,
    #         x='Player',
    #         y='SoT',
    #         color='Squad',
    #         title='Top 10 Shots on Target',
    #         template='plotly_dark'
    #     )
    #     fig.update_layout(
    #         height=400,
    #         xaxis_tickangle=45,
    #         plot_bgcolor='rgba(0,0,0,0)',
    #         paper_bgcolor='rgba(0,0,0,0)'
    #     )
    #     st.plotly_chart(fig)

# def create_team_stats(df):
#     """Create team statistics analysis"""
#     st.markdown("### Team Analysis")
    
#     # Ensure numeric columns
#     df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce').fillna(0)
#     df['90s'] = pd.to_numeric(df['90s'], errors='coerce').fillna(0)
#     df['SoT'] = pd.to_numeric(df['SoT'], errors='coerce').fillna(0)
    
#     # Calculate team-level statistics
#     team_stats = df.groupby('Squad').agg({
#         'Gls': 'sum',
#         'SoT': 'sum',
#         '90s': 'sum'
#     }).reset_index()
    
#     # Create visualization
#     fig = go.Figure()
    
#     fig.add_trace(go.Bar(
#         name='Goals',
#         x=team_stats['Squad'],
#         y=team_stats['Gls']
#     ))
    
#     fig.add_trace(go.Bar(
#         name='Shots on Target',
#         x=team_stats['Squad'],
#         y=team_stats['SoT']
#     ))
    
#     fig.update_layout(
#         title='Team Performance Overview',
#         barmode='group',
#         height=500,
#         xaxis_tickangle=45,
#         template='plotly_dark',
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)'
#     )
    
#     st.plotly_chart(fig)


def create_radar_chart(input_values, feature_names):
    """Create a radar chart of player attributes"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=input_values,
        theta=feature_names,
        fill='toself',
        name='Player Stats',
        line=dict(color='rgb(31, 119, 180)', width=2),
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        title={
            'text': "Player Attribute Profile",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=500,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

def create_confidence_bar_chart(probabilities, positions):
    """Create a bar chart of prediction probabilities"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=positions,
        y=probabilities,
        text=[f"{p:.1%}" for p in probabilities],
        textposition='auto',
        marker_color='rgb(31, 119, 180)',
        hoverinfo='y',
    ))
    
    fig.update_layout(
        title={
            'text': "Position Prediction Confidence",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Position",
        yaxis_title="Confidence",
        yaxis_range=[0, 1],
        yaxis_tickformat='.0%',
        showlegend=False,
        height=500,
        margin=dict(l=80, r=80, t=100, b=80),
        bargap=0.3
    )
    
    return fig

def create_position_analysis(df):
    """Create position distribution analysis"""
    st.markdown("### Player Position Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        position_counts = df['Position'].value_counts()
        fig = px.pie(
            values=position_counts.values,
            names=position_counts.index,
            title='Position Distribution in the League',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig)
    
    with col2:
        position_by_club = pd.crosstab(df['Squad'], df['Position'])
        position_by_club_pct = position_by_club.div(position_by_club.sum(axis=1), axis=0)
        
        fig = px.bar(
            position_by_club_pct,
            title='Position Distribution by Club',
            barmode='stack',
            template='plotly_dark'
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig)

def main():
    # Set page config
    st.set_page_config(
        page_title="Football Analytics Platform",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("Football Analytics Platform")
    
    # Create main navigation
    main_option = st.radio(
        "Select Analysis Type",
        ["Position Prediction", "Match Outcome Prediction"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if main_option == "Position Prediction":
        # Create tabs for position-related analysis
        position_tabs = st.tabs(["Player Position Predictor", "League Analysis", "Performance Metrics"])
        
        with position_tabs[0]:
             st.header("Player Position Predictor")
             
             # Load model and data
             model_data = load_model_position()
             pipeline = model_data['pipeline']
             feature_names = model_data['feature_names']
             feature_descriptions = model_data['feature_descriptions']
             
             # Create input form
             col1, col2 = st.columns(2)
             input_values = {}
             
             for i, feature in enumerate(feature_names):
                 description = feature_descriptions[feature]
                 if i % 2 == 0:
                     input_values[feature] = col1.number_input(
                         f"{description}",
                         min_value=0.0,
                         help=f"Enter the value for {description}"
                     )
                 else:
                     input_values[feature] = col2.number_input(
                         f"{description}",
                         min_value=0.0,
                         help=f"Enter the value for {description}"
                     )
             
             if st.button("Predict Position"):
                 X = pd.DataFrame([input_values])
                 prediction = pipeline.predict(X)[0]
                 probabilities = pipeline.predict_proba(X)[0]
                 
                 st.subheader("Prediction Results")
                 st.write(f"**Predicted Position:** {prediction}")
                 
                 # Display charts
                 chart_col1, chart_col2 = st.columns(2)
                 
                 # Confidence bar chart
                 positions = pipeline.classes_
                 fig_confidence = create_confidence_bar_chart(probabilities, positions)
                 chart_col1.plotly_chart(fig_confidence, use_container_width=True)
                 
                 # Radar chart
                 scaler = pipeline.named_steps['imputer']
                 normalized_values = scaler.transform(X)[0]
                 fig_radar = create_radar_chart(normalized_values, feature_names)
                 chart_col2.plotly_chart(fig_radar, use_container_width=True)
                 
                 # Display insights
                 st.subheader("Analysis Insights")
                 top_3_indices = np.argsort(probabilities)[-3:][::-1]
                 top_3_positions = [(positions[i], probabilities[i]) for i in top_3_indices]
                 
                 for pos, prob in top_3_positions:
                     st.write(f"- {pos}: {prob:.1%}")
                 
                 # Position-specific insights
                 st.write("\n**Key Insights:**")
                 if prediction == "FW":
                     st.write("- High attacking third presence")
                     st.write("- Strong progressive carries")
                 elif prediction == "MF":
                     st.write("- Balanced actions across all thirds")
                     st.write("- High number of progressive actions")
                 elif prediction == "DF":
                     st.write("- Strong defensive third presence")
                     st.write("- High number of clearances")
                 else:  # GK
                     st.write("- Limited outfield actions")
                     st.write("- High defensive penalty area presence")
        
        with position_tabs[1]:
             st.header("League Analysis")
             df = load_and_preprocess_data()
             
             # Display metrics
             col1, col2, col3 = st.columns(3)
             with col1:
                 st.metric("Total Players", f"{len(df):,}")
             with col2:
                 avg_defensive = df['Def Pen_Possession'].mean()
                 st.metric("Avg Defensive Actions", f"{avg_defensive:.1f}")
             with col3:
                 avg_progressive = df['PrgR_Possession'].mean()
                 st.metric("Avg Progressive Carries", f"{avg_progressive:.1f}")
             
             create_position_analysis(df)
             create_performance_analysis(df)
             
        with position_tabs[2]:
            st.header("Performance Metrics")
            df = load_and_preprocess_data()
            # create_performance_metrics(df)
            # create_team_stats(df)
    
    else:  # Match Outcome Prediction
      
        match_prediction_interface()
        

if __name__ == "__main__":
    main()