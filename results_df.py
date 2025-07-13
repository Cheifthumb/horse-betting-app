import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load and prepare data
data = pd.read_csv('Betting_Simulation/predicted_win_probabilities.M_24-25.csv')
data['Place'] = pd.to_numeric(data['Place'], errors='coerce')
data['Race_ID'] = data['Date of Race'].astype(str) + '_' + data['Time'].astype(str)
data['Odds_To_Use'] = data['Industry SP']
data['Predicted_Win_Probability'] = data.groupby('Race_ID')['Predicted_Win_Probability'].transform(lambda x: x / x.sum())
data['Expected_Value'] = (data['Predicted_Win_Probability'] * (data['Odds_To_Use'] - 1)) - (1 - data['Predicted_Win_Probability'])

# ✅ Parameter grid
ev_thresholds = [0.01, 0.02, 0.03]
kelly_fractions = [0.1, 0.2, 0.3]
max_odds_values = [30, 50, 75, 100]
bankroll_perc = 0.25

# ✅ Collect results
results = []

# ✅ Loop over each track
for track in data['Track'].dropna().unique():
    track_data = data[data['Track'] == track].copy()

    for ev_thresh, kelly_thresh, max_odds in itertools.product(ev_thresholds, kelly_fractions, max_odds_values):
        current_bankroll = 1000
        track_data = track_data.sort_values(by=['Date of Race', 'Time']).reset_index(drop=True)
        updated_rows = []

        for race_id, race_df in track_data.groupby('Race_ID', sort=False):
            stake_pool = current_bankroll * bankroll_perc
            race_df = race_df.copy()

            b = race_df['Odds_To_Use'] - 1
            p = race_df['Predicted_Win_Probability']
            q = 1 - p
            race_df['Kelly_Fraction'] = ((b * p) - q) / b

            race_df['Bet_Placed'] = (
                (race_df['Kelly_Fraction'] > kelly_thresh) &
                (race_df['Expected_Value'] > ev_thresh) &
                (race_df['Odds_To_Use'] <= max_odds)
            )

            race_df['Stake'] = 0
            race_df.loc[race_df['Bet_Placed'], 'Stake'] = race_df.loc[race_df['Bet_Placed'], 'Kelly_Fraction'] * stake_pool

            total_stake = race_df['Stake'].sum()
            if total_stake > stake_pool and total_stake > 0:
                race_df.loc[race_df['Bet_Placed'], 'Stake'] *= stake_pool / total_stake

            race_df['Actual_Result'] = (race_df['Place'] == 1).astype(int)
            race_df['Bet_Return'] = 0
            win_mask = race_df['Bet_Placed'] & (race_df['Actual_Result'] == 1)
            lose_mask = race_df['Bet_Placed'] & (race_df['Actual_Result'] == 0)

            race_df.loc[win_mask, 'Bet_Return'] = (race_df.loc[win_mask, 'Odds_To_Use'] - 1) * race_df.loc[win_mask, 'Stake']
            race_df.loc[lose_mask, 'Bet_Return'] = -race_df.loc[lose_mask, 'Stake']

            current_bankroll += race_df['Bet_Return'].sum()
            race_df['R_Multiple'] = 0
            race_df.loc[race_df['Bet_Placed'] & (race_df['Stake'] > 0), 'R_Multiple'] = race_df['Bet_Return'] / race_df['Stake']

            updated_rows.append(race_df)

        full_track = pd.concat(updated_rows)
        bets_placed = full_track['Bet_Placed'].sum()
        total_R = full_track.loc[full_track['Bet_Placed'], 'R_Multiple'].sum()
        winning_R = full_track.loc[full_track['R_Multiple'] > 0, 'R_Multiple'].sum()
        losing_R = full_track.loc[full_track['R_Multiple'] < 0, 'R_Multiple'].sum()

        results.append({
            'Track': track,
            'EV_Threshold': ev_thresh,
            'Min_Kelly': kelly_thresh,
            'Max_Odds': max_odds,
            'Total_Bets': bets_placed,
            'Total_R': total_R,
            'Winning_R': winning_R,
            'Losing_R': losing_R
        })

# ✅ Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['Track', 'Total_R'], ascending=[True, False])
results_df.to_excel('betting_simulation/trackwise_r_optimization.xlsx', index=False)

# ✅ Visualize top parameters
best_per_track = results_df.sort_values('Total_R', ascending=False).groupby('Track').head(1)

# ✅ Plot 1: Total R per Track
plt.figure(figsize=(12, 6))
sns.barplot(data=best_per_track.sort_values('Total_R', ascending=False), x='Track', y='Total_R', palette='viridis')
plt.title('📈 Best Total R per Track (Optimal Parameters)')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Total R')
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# ✅ Plot 2: Heatmap of EV Thresholds
pivot = best_per_track.pivot(index='Track', columns='EV_Threshold', values='Total_R')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap='YlGnBu')
plt.title('🔍 Total R by EV Threshold per Track')
plt.xlabel('EV Threshold')
plt.ylabel('Track')
plt.tight_layout()
plt.show()
