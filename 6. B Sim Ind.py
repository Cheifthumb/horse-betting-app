import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load predictions
data = pd.read_csv('Betting_Simulation/predicted_win_probabilities_test_24-25.csv')
data = data.copy()
data['Place'] = pd.to_numeric(data['Place'], errors='coerce')

# ✅ Sort chronologically
data = data.sort_values(by=['Date of Race', 'Time']).reset_index(drop=True)
data['Race_ID'] = data['Date of Race'].astype(str) + "_" + data['Time'].astype(str)

# ✅ Use Industry SP odds
data['Odds_To_Use'] = data['Industry SP']

# 🧠 Normalize predicted win probabilities to sum to 1.0 per race
data['Predicted_Win_Probability'] = data.groupby('Race_ID')['Predicted_Win_Probability'].transform(
    lambda x: x / x.sum()
)

# ✅ Calculate Expected Value using normalized probabilities
data['Expected_Value'] = (
    data['Predicted_Win_Probability'] * (data['Odds_To_Use'] - 1)
    - (1 - data['Predicted_Win_Probability'])
)

# ✅ Simulation settings
initial_bankroll = 10000
current_bankroll = initial_bankroll
bankroll_perc = 0.10         # 10% of bankroll per race
min_ev_threshold = 1      # Only bet when EV > 0.01
updated_rows = []

# ✅ Simulate race-by-race
for race_id, race_df in data.groupby('Race_ID', sort=False):
    num_horses = len(race_df)
    min_prob_threshold = 1 / num_horses
    stake_pool = current_bankroll * bankroll_perc

    # 📌 Bet only if both filters pass: prob threshold & EV threshold
    race_df['Bet_Placed'] = (
        (race_df['Predicted_Win_Probability'] > min_prob_threshold) &
        (race_df['Expected_Value'] > min_ev_threshold)
    )

    # 💸 Stake = stake_pool × predicted probability (only for placed bets)
    race_df['Stake'] = 0
    race_df.loc[race_df['Bet_Placed'], 'Stake'] = (
        race_df.loc[race_df['Bet_Placed'], 'Predicted_Win_Probability'] * stake_pool
    )

    # 🎯 Determine actual result
    race_df['Actual_Result'] = (race_df['Place'] == 1).astype(int)

    # 💰 Calculate returns: (SP - 1) × stake if win, -stake if loss
    race_df['Bet_Return'] = 0
    win_mask = (race_df['Bet_Placed']) & (race_df['Actual_Result'] == 1)
    lose_mask = (race_df['Bet_Placed']) & (race_df['Actual_Result'] == 0)

    race_df.loc[win_mask, 'Bet_Return'] = (
        (race_df.loc[win_mask, 'Odds_To_Use'] - 1) * race_df.loc[win_mask, 'Stake']
    )
    race_df.loc[lose_mask, 'Bet_Return'] = -race_df.loc[lose_mask, 'Stake']

    # 🏦 Update bankroll
    race_return = race_df['Bet_Return'].sum()
    current_bankroll += race_return
    race_df['Bankroll_After_Race'] = current_bankroll

    updated_rows.append(race_df)

# ✅ Combine all race data
data = pd.concat(updated_rows).reset_index(drop=True)

# ✅ Summary metrics
total_bets = data['Bet_Placed'].sum()
total_staked = data.loc[data['Bet_Placed'], 'Stake'].sum()
total_profit = data['Bet_Return'].sum()
roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
final_bankroll = current_bankroll
average_ev = data.loc[data['Bet_Placed'], 'Expected_Value'].mean()

# ✅ Print results
print("✅ Total Bets Placed:", int(total_bets))
print("✅ Total Amount Staked: ${:.2f}".format(total_staked))
print("✅ Total Profit/Loss: ${:.2f}".format(total_profit))
print("🏦 Final Bankroll: ${:.2f}".format(final_bankroll))
print("✅ ROI: {:.2f}%".format(roi))
print("✅ Average EV of Bets Placed: {:.4f}".format(average_ev))

# ✅ Save simulation results
data.to_csv('betting_simulation/betting_simulation_industry_dynamic_results_test_24-25.csv', index=False)

# ✅ Create DateTime column for plotting
data['Race_DateTime'] = pd.to_datetime(data['Date of Race'] + ' ' + data['Time'])

# ✅ Plot bankroll over time
plt.figure(figsize=(12,6))
plt.plot(data['Race_DateTime'], data['Bankroll_After_Race'], label='Bankroll', marker='o', markersize=2, linewidth=1)
plt.axhline(initial_bankroll, color='grey', linestyle='--', label='Starting Bankroll')
plt.title('📈 Bankroll Over Time (Normalized EV-Filtered)')
plt.xlabel('Race Date and Time')
plt.ylabel('Bankroll ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
