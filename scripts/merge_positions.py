import pandas as pd
import os

# File paths
log_file = "data/player_logs_2023-24.csv"
position_file = "data/2023-2024 NBA Player Stats - Playoffs.csv"
output_file = "data/player_logs_with_position_2023-2024.csv"

# Load player logs
print("📥 Loading player log data...")
logs_df = pd.read_csv(log_file)

# Load position file with correct delimiter
print("📥 Loading position file...")
positions_df = pd.read_csv(position_file, sep=";")

# Show column names to verify correct parsing
print("\n📋 Columns in position file:")
print(positions_df.columns.tolist())

# Auto-detect player and position column names
player_col = next((col for col in positions_df.columns if "player" in col.lower()), None)
position_col = next((col for col in positions_df.columns if "pos" in col.lower()), None)

if not player_col or not position_col:
    raise ValueError("❌ Could not detect 'PLAYER_NAME' or 'POSITION' columns. Please check the column names in the position file.")

# Rename for consistency
positions_df.rename(columns={player_col: "PLAYER_NAME", position_col: "POSITION"}, inplace=True)
positions_df = positions_df[["PLAYER_NAME", "POSITION"]].drop_duplicates()

# Merge based on name
print("🔀 Merging player positions...")
merged_df = logs_df.merge(positions_df, on="PLAYER_NAME", how="left")

# Report missing
missing_positions = merged_df["POSITION"].isna().sum()
print(f"\n⚠️ Missing positions for {missing_positions} player log rows.")

# Save
merged_df.to_csv(output_file, index=False)
print(f"✅ Done! Saved merged file as '{output_file}'")


