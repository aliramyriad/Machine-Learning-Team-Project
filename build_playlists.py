"""Stage 2: decide which songs are liked and which playlists each one fits.

For every song with enough history, asks the trained model "how likely is this
song to be liked if it came on in the morning / afternoon / evening / late at
night?". To answer that, it takes real listening moments from that time of day
(their hour, weekday, device, how the play started, how the session was going),
drops the song into each one, and averages the model's predictions.

A song is "liked" when its overall probability is at least LIKED_THRESHOLD. It
fits a playlist when you're predicted to like it more than a typical song at
that time of day (that context's real like rate + FIT_MARGIN). Each playlist is
ordered by how much more the song is liked at that time than at other times, so
the playlists aren't all the same favorites.

Usage:
    python build_playlists.py   (run train_model.py first)
"""
from pathlib import Path

import joblib
import pandas as pd

from features import CONTEXTS, FEATURES, build_features, history_snapshot, load_plays

MIN_PLAYS = 3          # songs heard fewer times than this are too unknown to judge
LIKED_THRESHOLD = 0.5
FIT_MARGIN = 0.05
PLAYLIST_SIZE = 50
MOMENTS_PER_CONTEXT = 100

# Features describing the listening moment rather than the song.
MOMENT_COLS = [
    "context", "hour", "weekday", "is_weekend", "platform", "reason_start",
    "shuffle", "offline", "session_pos", "session_like_rate", "prev_liked", "gap_min",
]


def candidate_rows(plays, tracks):
    snap = history_snapshot(plays)
    moments = pd.concat([
        plays.loc[plays["context"] == ctx, MOMENT_COLS].sample(MOMENTS_PER_CONTEXT, random_state=0)
        for ctx in CONTEXTS
    ])
    rows = tracks[["track_uri", "artist"]].merge(moments, how="cross")

    rows = rows.join(snap["track"], on="track_uri").join(snap["artist"], on="artist")
    rows = rows.join(snap["track_ctx"], on=["track_uri", "context"])
    rows = rows.join(snap["artist_ctx"], on=["artist", "context"])
    rows = rows.join(snap["days_since_first_play"], on="track_uri")
    # Never heard in this context yet -> zero plays, neutral rate.
    for name in ("track_ctx", "artist_ctx"):
        rows[f"{name}_prev_plays"] = rows[f"{name}_prev_plays"].fillna(0)
        rows[f"{name}_like_rate"] = rows[f"{name}_like_rate"].fillna(0.5)
    return rows


def main():
    model = joblib.load("models/like_model.joblib")
    plays = build_features(load_plays())

    tracks = (plays.groupby("track_uri")
              .agg(track=("track", "first"), artist=("artist", "first"),
                   plays=("liked", "size"), past_like_rate=("liked", "mean"))
              .reset_index())
    tracks = tracks[tracks["plays"] >= MIN_PLAYS]

    rows = candidate_rows(plays, tracks)
    rows["p"] = model.predict_proba(rows[FEATURES])[:, 1]
    p = rows.groupby(["track_uri", "context"])["p"].mean().unstack()[list(CONTEXTS)]

    out = tracks.set_index("track_uri")
    # Overall chance of liking it, weighting each context by how much you listen then.
    context_share = plays["context"].value_counts(normalize=True)[list(CONTEXTS)]
    out["p_liked"] = (p * context_share).sum(axis=1)
    out["liked"] = out["p_liked"] >= LIKED_THRESHOLD
    context_like_rate = plays.groupby("context")["liked"].mean()[list(CONTEXTS)]
    lift = p - context_like_rate  # how much better than a typical song at that time
    for ctx in CONTEXTS:
        out[f"p_{ctx}"] = p[ctx]
        out[f"fits_{ctx}"] = p[ctx] >= context_like_rate[ctx] + FIT_MARGIN
    out = out.sort_values("p_liked", ascending=False).round(3)

    Path("outputs/playlists").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/track_predictions.csv")
    print(f"Scored {len(out):,} songs with {MIN_PLAYS}+ plays: "
          f"{out['liked'].sum():,} predicted liked.\n")

    for ctx in CONTEXTS:
        fits = out[out[f"fits_{ctx}"]]
        preference = lift.loc[fits.index, ctx] - lift.loc[fits.index].mean(axis=1)
        playlist = (fits.assign(preference=preference.round(3))
                    .sort_values("preference", ascending=False)
                    .head(PLAYLIST_SIZE)[["track", "artist", f"p_{ctx}", "preference"]])
        playlist.to_csv(f"outputs/playlists/{ctx}.csv")
        print(f"{ctx.replace('_', ' ').title()} playlist: "
              f"{out[f'fits_{ctx}'].sum():,} songs fit, top 5:")
        print(playlist.head(5).to_string(index=False), "\n")
    print("Wrote outputs/track_predictions.csv and outputs/playlists/*.csv")


if __name__ == "__main__":
    main()
