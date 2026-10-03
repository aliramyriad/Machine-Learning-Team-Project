"""Turn a raw Spotify Extended Streaming History export into data/plays.csv.

Keeps only music plays and the fields the model needs. Drops personal info
(IP address etc.) so the cleaned file is safe to commit.

Usage:
    python prepare_data.py path/to/Streaming_History_Audio_*.json [more files...]
"""
import sys
from pathlib import Path

import pandas as pd

KEEP = {
    "ts": "ts",
    "ms_played": "ms_played",
    "master_metadata_track_name": "track",
    "master_metadata_album_artist_name": "artist",
    "master_metadata_album_album_name": "album",
    "spotify_track_uri": "track_uri",
    "platform": "platform",
    "reason_start": "reason_start",
    "reason_end": "reason_end",
    "shuffle": "shuffle",
    "skipped": "skipped",
    "offline": "offline",
}


def main(paths):
    raw = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
    plays = raw[raw["spotify_track_uri"].notna()][list(KEEP)].rename(columns=KEEP)
    plays = plays.drop_duplicates().sort_values("ts").reset_index(drop=True)

    out = Path("data/plays.csv")
    out.parent.mkdir(exist_ok=True)
    plays.to_csv(out, index=False)
    print(f"Wrote {len(plays):,} plays ({plays.track_uri.nunique():,} tracks) to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
