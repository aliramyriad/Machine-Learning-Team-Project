"""Labels and features for the "will this play be liked?" model.

Every feature for a play is computed only from what happened *before* that
play (running counts, shifted by one), so the model never sees the outcome it
is predicting.
"""
import json

import pandas as pd

LOCAL_TZ = "America/New_York"
SESSION_GAP_MIN = 30   # a gap longer than this starts a new listening session
PRIOR_RATE = 0.5       # smoothing: songs with few plays are pulled toward 50% liked
PRIOR_WEIGHT = 2

# Playlist contexts, by local hour the song started.
CONTEXTS = {
    "morning": range(5, 12),
    "afternoon": range(12, 17),
    "evening": range(17, 22),
    "late_night": [22, 23, 0, 1, 2, 3, 4],
}
HOUR_TO_CONTEXT = {h: ctx for ctx, hours in CONTEXTS.items() for h in hours}

# Song audio features from the ReccoBeats API (exported to track_dataset.json).
AUDIO = [
    "energy", "danceability", "tempo", "valence", "acousticness", "instrumentalness",
    "speechiness", "liveness", "loudness", "key", "mode",
]

CATEGORICAL = ["platform", "reason_start", "context"]
NUMERIC = [
    "hour", "weekday", "is_weekend", "shuffle", "offline",
    "track_prev_plays", "track_like_rate",
    "artist_prev_plays", "artist_like_rate",
    "track_ctx_prev_plays", "track_ctx_like_rate",
    "artist_ctx_prev_plays", "artist_ctx_like_rate",
    "days_since_first_play",
    "session_pos", "session_like_rate", "prev_liked", "gap_min",
]
FEATURES_NO_AUDIO = CATEGORICAL + NUMERIC
FEATURES = FEATURES_NO_AUDIO + AUDIO

# (feature prefix, grouping columns) for the running like-rate features
HISTORY_GROUPS = [
    ("track", ["track_uri"]),
    ("artist", ["artist"]),
    ("track_ctx", ["track_uri", "context"]),
    ("artist_ctx", ["artist", "context"]),
]


def smoothed_rate(likes, plays):
    return (likes + PRIOR_RATE * PRIOR_WEIGHT) / (plays + PRIOR_WEIGHT)


def load_plays(path="data/plays.csv"):
    df = pd.read_csv(path, parse_dates=["ts"])
    df["artist"] = df["artist"].fillna("unknown")
    return df.sort_values("ts").reset_index(drop=True)


def load_audio_features(path="track_dataset.json"):
    """Audio features per track, indexed by track URI.

    Only the audio features are used: the file's play_count / minutes /
    first_played columns summarise the whole year, test months included.
    """
    with open(path) as f:
        records = json.load(f)
    audio = pd.DataFrame([{"track_uri": r["spotify_track_uri"], **r["audio_features"]}
                          for r in records])
    return audio.set_index("track_uri")[AUDIO]


def add_label_and_context(df):
    df = df.copy()
    # "Liked" = listened past 30s without skipping (Spotify exports have no hearts).
    df["liked"] = (~df["skipped"].astype(bool) & (df["ms_played"] >= 30_000)).astype(int)

    # ts is when the play *ended*; work out when it started, in local time.
    df["end"] = df["ts"].dt.tz_convert(LOCAL_TZ)
    df["start"] = df["end"] - pd.to_timedelta(df["ms_played"], unit="ms")
    df["hour"] = df["start"].dt.hour
    df["weekday"] = df["start"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["context"] = df["hour"].map(HOUR_TO_CONTEXT)
    df["shuffle"] = df["shuffle"].astype(int)
    df["offline"] = df["offline"].astype(int)
    return df


def build_features(df):
    """Add the leak-free feature columns to a plays table."""
    df = add_label_and_context(df)

    for name, keys in HISTORY_GROUPS:
        g = df.groupby(keys)["liked"]
        prev_plays = g.cumcount()
        prev_likes = g.cumsum() - df["liked"]
        df[f"{name}_prev_plays"] = prev_plays
        df[f"{name}_like_rate"] = smoothed_rate(prev_likes, prev_plays)

    first_start = df.groupby("track_uri")["start"].transform("min")
    df["days_since_first_play"] = (df["start"] - first_start).dt.total_seconds() / 86_400

    # Listening sessions: consecutive plays with short gaps between them.
    gap = (df["start"] - df["end"].shift()).dt.total_seconds() / 60
    df["session_id"] = (gap.isna() | (gap > SESSION_GAP_MIN)).cumsum()
    s = df.groupby("session_id")["liked"]
    df["session_pos"] = s.cumcount()
    df["session_like_rate"] = smoothed_rate(s.cumsum() - df["liked"], df["session_pos"])
    df["prev_liked"] = s.shift()  # NaN on the first song of a session
    df["gap_min"] = gap.where(df["session_pos"] > 0)

    # Songs missing from the audio dataset get NaN features.
    return df.join(load_audio_features(), on="track_uri")


def history_snapshot(df):
    """End-of-history stats per track, artist and context, for scoring songs now.

    `df` must already have gone through add_label_and_context.
    """
    snap = {}
    for name, keys in HISTORY_GROUPS:
        agg = df.groupby(keys)["liked"].agg(["count", "sum"])
        snap[name] = pd.DataFrame({
            f"{name}_prev_plays": agg["count"],
            f"{name}_like_rate": smoothed_rate(agg["sum"], agg["count"]),
        })
    first_start = df.groupby("track_uri")["start"].min()
    snap["days_since_first_play"] = (
        (df["end"].max() - first_start).dt.total_seconds() / 86_400
    ).rename("days_since_first_play")
    return snap
