import pandas as pd
import time
import os
import requests
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog
from nba_api.stats.library.parameters import SeasonTypeAllStar

# ------------------------------------------
# ⚙️ Config
# ------------------------------------------
SEASON = '2023-24'
OUTPUT_DIR = 'data'
PLAYER_RETRIES = 3
TEAM_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1
RETRY_BACKOFF = 5

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load player and team metadata
all_players = players.get_active_players()
all_teams = teams.get_teams()

# ------------------------------------------
# 🔁 Player Logs
# ------------------------------------------
def get_player_logs(season):
    logs = []
    for player in all_players:
        pid = player['id']
        name = player['full_name']
        attempt = 0
        success = False

        while attempt < PLAYER_RETRIES and not success:
            try:
                print(f"📥 Fetching {name} ({pid}) - Attempt {attempt+1}")
                log = playergamelog.PlayerGameLog(
                    player_id=pid,
                    season=season,
                    season_type_all_star=SeasonTypeAllStar.regular
                )
                df = log.get_data_frames()[0]
                df['PLAYER_ID'] = pid
                df['PLAYER_NAME'] = name
                df['SEASON'] = season
                logs.append(df)
                success = True
            except requests.exceptions.ReadTimeout:
                print(f"⏳ Timeout for {name}. Retrying in {RETRY_BACKOFF}s...")
                attempt += 1
                time.sleep(RETRY_BACKOFF)
            except Exception as e:
                print(f"❌ Failed for {name}: {e}")
                break
        time.sleep(SLEEP_BETWEEN_CALLS)
    return pd.concat(logs, ignore_index=True)

# ------------------------------------------
# 🔁 Team Logs
# ------------------------------------------
def get_team_logs(season):
    logs = []
    for team in all_teams:
        tid = team['id']
        name = team['full_name']
        attempt = 0
        success = False

        while attempt < TEAM_RETRIES and not success:
            try:
                print(f"🏀 Fetching {name} ({tid}) - Attempt {attempt+1}")
                log = teamgamelog.TeamGameLog(
                    team_id=tid,
                    season=season,
                    season_type_all_star=SeasonTypeAllStar.regular
                )
                df = log.get_data_frames()[0]
                df['TEAM_ID'] = tid
                df['TEAM_NAME'] = name
                df['SEASON'] = season
                logs.append(df)
                success = True
            except requests.exceptions.ReadTimeout:
                print(f"⏳ Timeout for {name}. Retrying in {RETRY_BACKOFF}s...")
                attempt += 1
                time.sleep(RETRY_BACKOFF)
            except Exception as e:
                print(f"❌ Failed for {name}: {e}")
                break
        time.sleep(SLEEP_BETWEEN_CALLS)
    return pd.concat(logs, ignore_index=True)

# ------------------------------------------
# 🚀 Main
# ------------------------------------------
if __name__ == "__main__":
    print(f"🚀 Starting NBA data fetch for {SEASON} season...\n")

    try:
        player_df = get_player_logs(SEASON)
        player_df.to_csv(os.path.join(OUTPUT_DIR, f'player_logs_{SEASON}.csv'), index=False)
        print(f"\n✅ Player logs saved to data/player_logs_{SEASON}.csv")
    except Exception as e:
        print(f"❌ Failed to fetch player logs: {e}")

    try:
        team_df = get_team_logs(SEASON)
        team_df.to_csv(os.path.join(OUTPUT_DIR, f'team_logs_{SEASON}.csv'), index=False)
        print(f"\n✅ Team logs saved to data/team_logs_{SEASON}.csv")
    except Exception as e:
        print(f"❌ Failed to fetch team logs: {e}")

    print("\n🎉 Done.")
