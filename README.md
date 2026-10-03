# Spotify Song Classifier

Predicts whether a song will be **liked**, and which **playlists** it fits
(morning / afternoon / evening / late night), from one person's Spotify
listening history (Jan–Sep 2026, ~14.6k plays, ~4.9k songs).

## How it works

**Label.** Spotify exports don't include hearts, so a play counts as *liked*
when it wasn't skipped and lasted at least 30 seconds (44.5% of plays).

**Stage 1 – "Will this song be liked?"** (`train_model.py`)
A classifier predicts whether each play will be liked. Every feature comes from
what happened *before* that play, so the model can't cheat:
- the song's and artist's past like rate and play count (overall and at this time of day)
- time of day, weekday, device, shuffle, how the play started
- how the current listening session is going (was the last song liked? etc.)

It trains on Jan–Jul and is tested on Aug–Sep:

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Always guess "not liked" | 0.51 | 0.00 | 0.50 |
| Logistic regression | 0.75 | 0.75 | 0.84 |
| Gradient boosting | 0.75 | 0.74 | 0.83 |

**Stage 2 – "Which playlist does it fit?"** (`build_playlists.py`)
For each song with 3+ plays, the model scores it in 100 real listening moments
from each time of day and averages the results. A song:
- is **liked** if its overall probability is ≥ 0.5
- **fits a playlist** if you're predicted to like it at least 5 points more than
  a typical song at that time of day

Each playlist is ordered by how much more the song is liked at that time than at
other times, so the four playlists don't just repeat the same favorites.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# only needed when starting from a new raw Spotify export:
python prepare_data.py path/to/Streaming_History_Audio_*.json

python train_model.py      # prints metrics, saves models/like_model.joblib
python build_playlists.py  # writes outputs/track_predictions.csv and outputs/playlists/*.csv
```

## Files

| File | Purpose |
|---|---|
| `data/plays.csv` | Cleaned plays (music only; IP address and other personal fields removed) |
| `features.py` | Label, time-of-day contexts, and leak-free feature engineering |
| `train_model.py` | Stage 1: train, evaluate, save model; writes `outputs/metrics.json` |
| `build_playlists.py` | Stage 2: liked verdict and playlist fit for every song |
| `spotifyML.py` | Early Spotify Web API test |

## Limitations / next steps

- **No workout playlist yet.** Spotify blocks the audio-features endpoint
  (energy, tempo, danceability) for new apps (it returns 403), and listening
  history alone can't tell when you were working out. To add workout/chill
  playlists, join a public dataset that already has audio features (e.g. a
  Kaggle Spotify tracks dataset) on track ID, add those columns as features,
  and define workout as high energy and tempo.
- The strongest signal is session "mood" (skip streaks); song-specific signal is
  weaker, so playlist scores for songs with few plays are uncertain.
- "Liked" is a proxy for real likes, so if you have actual liked-songs data
  (Spotify's account data export includes `YourLibrary.json`), use it as the label.
