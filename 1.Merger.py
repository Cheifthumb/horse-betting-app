import pandas as pd
import os
from datetime import datetime

# ==== USER INPUT ====
base_folder = "Years"
include_years = ['2023', '2024', '2025']
output_name = f"Selection_Merged_{'_'.join(include_years)}.xlsx"
sheet_name = "MergedRaces_10-22"
output_folder = "Merged"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, output_name)
# =====================

# Build full file paths like Years/2021.xlsx
excel_files = [
    os.path.join(base_folder, f"{year}.xlsx") for year in include_years
    if os.path.isfile(os.path.join(base_folder, f"{year}.xlsx"))
]

dfs = []
for file in excel_files:
    try:
        df = pd.read_excel(file, engine="openpyxl")
        df.columns = df.columns.str.strip()
        print(f"📄 Reading: {file}")
        print(f"🧾 Columns: {df.columns.tolist()}")
        if 'Date of Race' in df.columns and 'Time' in df.columns:
            dfs.append(df)
        else:
            print(f"⚠️ Skipping {file}: missing 'Date of Race' or 'Time'")
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# Combine and sort
if dfs:
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['Race Datetime'] = pd.to_datetime(
        full_df['Date of Race'].astype(str) + ' ' + full_df['Time'].astype(str),
        errors='coerce', dayfirst=True
    )
    full_df = full_df.dropna(subset=['Race Datetime'])
    full_df = full_df.sort_values(['Date of Race', 'Time']).reset_index(drop=True)

    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        full_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅ Merged data saved to: {output_file}")
    print("📊 Preview:\n", full_df.head())
else:
    print("❌ No valid files to merge.")
