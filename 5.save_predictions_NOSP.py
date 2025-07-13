import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load SP-free featured data
data = pd.read_excel('featured_data/16.05.2025.xlsx')

# Info columns to save (keep Betfair SP only for simulation, not model input)
info_columns = [
    'Date of Race', 'Time', 'Track', 'Horse', 'Distance',
    'Place', 'Betfair SP', 'Industry SP'
]

# SP-free pre-race features (aligned with training script)
pre_race_features = [
    'Going', 'Distance', 'Class', 'Stall', 'Official Rating', 'Age', 'Weight',
    'Runs last 18 months', 'Wins Last 5 races',
    'Avg % SP Drop Last 5 races', 'Avg % SP Drop last 18 mths',
    'RBD Rating', 'RBD Rank', 'Total Prev Races',
    'Days Since Last time out', 'Course Wins', 'Distance Wins', 'Class Wins', 'Going Wins',
    'Up in Trip'
]

# Filter valid features
existing_features = [f for f in pre_race_features if f in data.columns]
X = data[existing_features]

# Encode categoricals
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(-999)

# Load model
model = joblib.load('models/xgboost_tuned_model_02-25.pkl')  # ✅ Correct model file for this data

# Predict
win_probabilities = model.predict_proba(X)[:, 1]

# Output results
output = data[info_columns].copy()
output['Predicted_Win_Probability'] = win_probabilities

# Save to file
output.to_csv('Betting_Simulation/predicted_win_probabilities_16.05.2025.csv', index=False)
print("✅ Predictions saved with Betfair SP to 'Betting_Simulation/predicted_win_probabilities.csv'")
