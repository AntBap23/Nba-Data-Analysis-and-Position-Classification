# scripts/fetch_player_bio.py

from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd
import time

OUTPUT_FILE = "data/nba_player_bio.csv"

def fetch_player_bio(player_id):
    """Fetch player bio info from NBA API."""
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    return {
        "PLAYER_ID": player_id,
        "PLAYER_NAME": info.loc[0, "DISPLAY_FIRST_LAST"],
        "POSITION": info.loc[0, "POSITION"],
        "HEIGHT": info.loc[0, "HEIGHT"],
        "WEIGHT": info.loc[0, "WEIGHT"],
        "BIRTHDATE": info.loc[0, "BIRTHDATE"],
        "TEAM": info.loc[0, "TEAM_NAME"]
    }

if __name__ == "__main__":
    print("📥 Fetching active player list...")
    active_players = players.get_active_players()

    all_bios = []
    for i, p in enumerate(active_players, start=1):
        try:
            print(f"📌 {i}/{len(active_players)} - {p['full_name']}")
            bio = fetch_player_bio(p["id"])
            all_bios.append(bio)
        except Exception as e:
            print(f"❌ Failed for {p['full_name']}: {e}")
        time.sleep(0.6)  # avoid hitting rate limits

    df = pd.DataFrame(all_bios)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done! Saved {len(df)} player bios to {OUTPUT_FILE}")
