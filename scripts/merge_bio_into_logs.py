import os
import re
import pandas as pd

LOGS_CSV = "data/player_logs_2023-24.csv"
BIO_CSV  = "data/nba_player_bio.csv"
OUT_CSV  = "data/player_logs_2023-24_with_bio.csv"

def parse_height_to_inches(h):
    """Convert '6-8' or 6'8" to inches, else NaN."""
    if pd.isna(h): return float('nan')
    s = str(h).strip().replace('"', '').replace('’', "'").replace("`", "'")
    s = s.replace(" ", "").replace("–","-").replace("—","-")
    m_dash = re.fullmatch(r"(\d+)-(\d+)", s)
    m_tick = re.fullmatch(r"(\d+)'(\d+)", s)
    if m_dash:
        ft, inch = map(int, m_dash.groups()); return ft*12 + inch
    if m_tick:
        ft, inch = map(int, m_tick.groups()); return ft*12 + inch
    try:
        return float(s)
    except:
        return float('nan')

def collapse_duplicate_player_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple columns share the label 'PLAYER_ID', coalesce them into a single
    Int64 column by taking the first non-null numeric value across duplicates.
    Works even when duplicate labels make df['PLAYER_ID'] a DataFrame.
    """
    # Find the index positions of every 'PLAYER_ID' column
    idxs = [i for i, c in enumerate(df.columns) if c == "PLAYER_ID"]
    if len(idxs) <= 1:
        return df

    first_idx = idxs[0]
    # Start with the first 'PLAYER_ID' column (by position) as a Series
    base = pd.to_numeric(df.iloc[:, first_idx], errors="coerce")

    # Fill missing values from subsequent duplicate columns
    for j in idxs[1:]:
        base = base.fillna(pd.to_numeric(df.iloc[:, j], errors="coerce"))

    # Drop all PLAYER_ID columns (by label), then reinsert the coalesced one
    df = df.drop(columns=["PLAYER_ID"])
    df.insert(first_idx, "PLAYER_ID", base.astype("Int64"))

    return df

def main():
    # ---------- Load ----------
    logs = pd.read_csv(LOGS_CSV)
    bio  = pd.read_csv(BIO_CSV)

    # ---------- Normalize column names ----------
    logs.columns = [c.upper().strip() for c in logs.columns]
    bio.columns  = [c.upper().strip() for c in bio.columns]

    # If bios have DISPLAY_FIRST_LAST but not PLAYER_NAME, align it
    if "PLAYER_NAME" not in bio.columns and "DISPLAY_FIRST_LAST" in bio.columns:
        bio.rename(columns={"DISPLAY_FIRST_LAST":"PLAYER_NAME"}, inplace=True)

    # ---------- Fix duplicate columns / ensure keys exist ----------
    # Some logs have both Player_ID and PLAYER_ID; after upper, we may have dup labels.
    logs = collapse_duplicate_player_id_cols(logs)

    # If logs still don't have PLAYER_ID but have something close (e.g., PLAYERID), try to map
    if "PLAYER_ID" not in logs.columns:
        cand = next((c for c in logs.columns if c.replace("_","") == "PLAYERID"), None)
        if cand:
            logs.rename(columns={cand: "PLAYER_ID"}, inplace=True)

    # Ensure required keys are present
    if "PLAYER_ID" not in logs.columns:
        raise ValueError(f"'PLAYER_ID' not found in logs columns: {list(logs.columns)}")
    if "PLAYER_ID" not in bio.columns:
        raise ValueError(f"'PLAYER_ID' not found in bio columns: {list(bio.columns)}")
    if "PLAYER_NAME" not in logs.columns and "PLAYER_NAME" in bio.columns:
        # Make sure logs at least have a name column; if not, we can try to create from whatever exists
        # (Most nba_api logs have PLAYER_NAME already. If not, this will just skip name-based fallback.)
        pass

    # ---------- Parse/standardize physicals in bio ----------
    if "HEIGHT" in bio.columns:
        bio["HEIGHT_IN"] = bio["HEIGHT"].apply(parse_height_to_inches)
    if "WEIGHT" in bio.columns:
        bio["WEIGHT_LB"] = pd.to_numeric(bio["WEIGHT"], errors="coerce")

    # Avoid position collision if logs already have POSITION from somewhere else
    if "POSITION" in logs.columns and "POSITION" in bio.columns:
        bio.rename(columns={"POSITION": "BIO_POSITION"}, inplace=True)

    # Keep one row per player in bios
    keep_cols = ["PLAYER_ID"]
    for col in ["PLAYER_NAME","BIO_POSITION","POSITION","HEIGHT","HEIGHT_IN","WEIGHT","WEIGHT_LB","BIRTHDATE","TEAM","TEAM_NAME"]:
        if col in bio.columns:
            keep_cols.append(col)
    bio_one = bio[keep_cols].drop_duplicates(subset=["PLAYER_ID"])

    # ---------- Primary merge on PLAYER_ID ----------
    merged = logs.merge(bio_one, on="PLAYER_ID", how="left", suffixes=("", "_BIO"))

    # ---------- Create/resolve POSITION column ----------
    # If logs don't have POSITION, build it from bios (BIO_POSITION or POSITION from bio)
    if "POSITION" not in logs.columns:
        if "BIO_POSITION" in merged.columns:
            merged["POSITION"] = merged["BIO_POSITION"]
        elif "POSITION" in bio_one.columns:
            # If we didn’t rename bio POSITION, it’s already merged as POSITION
            pass
    else:
        # If logs had POSITION and bio also had BIO_POSITION, fill missing from BIO_POSITION
        if "BIO_POSITION" in merged.columns:
            merged["POSITION"] = merged["POSITION"].combine_first(merged["BIO_POSITION"])

    # ---------- Optional fallback: fill from name-based merge if still missing ----------
    need_fallback = False
    if "POSITION" in merged.columns:
        need_fallback = merged["POSITION"].isna().any()
    elif "HEIGHT_IN" in merged.columns:
        need_fallback = merged["HEIGHT_IN"].isna().any()

    if need_fallback and "PLAYER_NAME" in logs.columns and "PLAYER_NAME" in bio_one.columns:
        bio_by_name = bio_one.drop_duplicates(subset=["PLAYER_NAME"])
        merged = merged.merge(
            bio_by_name, on="PLAYER_NAME", how="left", suffixes=("", "_BYNAME")
        )
        # Fill fields from _BYNAME where missing
        for tgt, src in [
            ("POSITION","BIO_POSITION_BYNAME"),
            ("BIO_POSITION","BIO_POSITION_BYNAME"),
            ("HEIGHT_IN","HEIGHT_IN_BYNAME"),
            ("WEIGHT_LB","WEIGHT_LB_BYNAME"),
            ("BIRTHDATE","BIRTHDATE_BYNAME"),
            ("TEAM","TEAM_BYNAME"),
            ("TEAM_NAME","TEAM_NAME_BYNAME"),
        ]:
            if tgt in merged.columns and src in merged.columns:
                merged[tgt] = merged[tgt].combine_first(merged[src])
        # Drop helper cols
        drop_cols = [c for c in merged.columns if c.endswith("_BYNAME")]
        if drop_cols:
            merged.drop(columns=drop_cols, inplace=True)

    # Final tidy: if BIO_POSITION remains and we’ve already consolidated, drop it
    if "BIO_POSITION" in merged.columns:
        if "POSITION" not in merged.columns:
            merged.rename(columns={"BIO_POSITION":"POSITION"}, inplace=True)
        else:
            merged.drop(columns=["BIO_POSITION"], inplace=True)

    # ---------- Save ----------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)

    # ---------- Report ----------
    total = len(merged)
    pos_missing = merged["POSITION"].isna().sum() if "POSITION" in merged.columns else total
    print(f"✅ Saved: {OUT_CSV}")
    print(f"   Rows: {total:,}")
    print(f"   Missing POSITION: {pos_missing:,}")
    if "HEIGHT_IN" in merged.columns:
        print(f"   Missing HEIGHT_IN: {merged['HEIGHT_IN'].isna().sum():,}")
    if "WEIGHT_LB" in merged.columns:
        print(f"   Missing WEIGHT_LB: {merged['WEIGHT_LB'].isna().sum():,}")

if __name__ == "__main__":
    main()


