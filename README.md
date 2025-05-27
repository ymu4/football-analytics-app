# ⚽ Football Analytics Web App

This interactive Streamlit application combines two machine learning models into one seamless interface:

1. 🔍 [Player Position Classifier](https://github.com/ymu4/player-position-classifier)  
2. 🧠 [Match Result Predictor](https://github.com/ymu4/football-match-outcome-model-predictor
)

It allows users to upload, analyze, and predict football outcomes and player roles using real statistics.

---

## 🚀 Features

### 🧠 Player Position Classification
- Predicts football player roles: Defender, Midfielder, Forward, or Goalkeeper
- Trained on real 2023/2024 Premier League statistics
- Outputs predictions with confidence bars and radar charts

### 🎯 Match Outcome Prediction
- Predicts match result (home win vs not win) using pre-game features like:
  - Form, momentum, streaks, betting odds, clean sheet rate
- Includes advanced feature engineering and ensemble models (RF + XGB)

---

## 🧾 Project Structure

```
football-analytics-app/
├── app.py
├── football_enhanced2.py
├── match_outcome_predictor.py
├── prepare_data.py
├── filter_datase.py
├── *.joblib
├── *.xlsx / *.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py
```

---

## 🧠 Models Used

- RandomForestClassifier (for player classification)
- XGBoostClassifier + Logistic Meta (for match results)
- All models are saved using joblib and loaded dynamically

---

## 📊 Visual Outputs

- Position probability bars and radar charts
- Match win probabilities and confidence scores
- Feature importance plots and confusion matrices

---

## 📄 License

MIT © 2025 Sumaya Nasser Alhashmi

---

📦 Model Repositories:
- [🔗 Player Position Classifier](https://github.com/ymu4/player-position-classifier)
- [🔗 Match Result Predictor](https://github.com/ymu4/football-match-outcome-model-predictor
)
