import pandas as pd

data = pd.read_csv('featured_data/predicted_win_probabilities.csv')

print("\n✅ Columns in dataset:", list(data.columns))

print("\n✅ Quick sample of data:")
print(data.head())

print("\n✅ Number of horses who actually WON:")
print((data['Place'] == 1).sum())

print("\n✅ Betting threshold used (Predicted Win Probability > 0.20):")
print("Number of horses above threshold:", (data['Predicted_Win_Probability'] > 0.20).sum())

print("\n✅ Any horses where Industry SP is missing?")
print(data['Industry SP'].isnull().sum())

print("\n✅ Minimum and Maximum values of Industry SP:")
print(data['Industry SP'].min(), data['Industry SP'].max())
