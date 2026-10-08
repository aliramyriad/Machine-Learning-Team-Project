# Spotify Song Classifier

Predicts whether a song will be **liked**, and which **playlists** it fits
(morning / afternoon / evening / late night / workout / chill), from one
person's Spotify listening history (Jan–Sep 2026, ~14.6k plays, ~4.9k songs)
plus song audio features from the [ReccoBeats API](https://reccobeats.com).

## How it works

**Label.** Spotify exports don't include hearts, so a play counts as *liked*
when it wasn't skipped and lasted at least 30 seconds (44.5% of plays).

**Stage 1 – "Will this song be liked?"** (`train_model.py`)
A classifier predicts whether each play will be liked. Every feature comes from
what happened *before* that play, so the model can't cheat:
- the song's and artist's past like rate and play count (overall and at this time of day)
- time of day, weekday, device, shuffle, how the play started
- how the current listening session is going (was the last song liked? etc.)
- the song's audio features from ReccoBeats: energy, danceability, tempo,
  valence, acousticness, instrumentalness, speechiness, liveness, loudness,
  key, mode (available for 73% of plays; missing values are left blank)

It trains on Jan–Jul and is tested on Aug–Sep, with and without audio features:

| Model | Accuracy | F1 | ROC-AUC | Training time (s)|
|---|---|---|---|---|
| Always guess "not liked" | 0.51 | 0.00 | 0.50 | 0.001 |
| Logistic regression, no audio | 0.75 | 0.75 | 0.835 | 0.038 |
| Logistic regression + audio | 0.75 | 0.75 | 0.837 | 0.058 |
| Gradient boosting, no audio | 0.75 | 0.74 | 0.828 | 0.312 |
| Gradient boosting + audio | 0.75 | 0.74 | 0.828 | 0.356 |
| Neural network, no audio features | 0.76 | 0.75 | 0.838 | 0.780 |
| Neural network + audio features | 0.76 | 0.75 | 0.832 | 0.821 |

Audio features barely help: past skips of the song and artist already capture
taste, and the strongest signal is session "mood" (was the last song liked?).

The file's `play_count` / minutes / `first_played` / `last_played` columns are
deliberately **not** used: they summarise the whole year, including the test
months, so they would leak the answer.

**Stage 2 – "Which playlist does it fit?"** (`build_playlists.py`)
For each song with 3+ plays, the model scores it in 100 real listening moments
from each time of day and averages the results. A song:
- is **liked** if its overall probability is ≥ 0.5
- **fits a playlist** if you're predicted to like it at least 5 points more than
  a typical song at that time of day

Each playlist is ordered by how much more the song is liked at that time than at
other times, so the four playlists don't just repeat the same favorites.

**Workout** and **Chill** are rule-based on audio features, because there's no
record of when you were working out to learn from. They take the liked songs
whose sound matches (thresholds in `SOUND_PLAYLISTS` in `build_playlists.py`):
- Workout: energy ≥ 0.65, danceability ≥ 0.6, tempo ≥ 100 BPM
- Chill: energy ≤ 0.45, acousticness ≥ 0.4

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
| `track_dataset.json` | Per-song audio features from the ReccoBeats API |
| `features.py` | Label, time-of-day contexts, and leak-free feature engineering |
| `train_model.py` | Stage 1: train, evaluate, save model; writes `outputs/metrics.json` |
| `build_playlists.py` | Stage 2: liked verdict and playlist fit for every song |
| `spotifyML.py` | Early Spotify Web API test |

## Limitations / next steps

- Spotify's own audio-features endpoint returns 403 for new apps, so audio
  features come from ReccoBeats instead. 27% of plays (mostly songs played
  once) have no audio features, and those songs can't appear in Workout/Chill.
- Workout and Chill thresholds are hand-picked, not learned.
- The strongest signal is session "mood" (skip streaks); song-specific signal is
  weaker, so playlist scores for songs with few plays are uncertain.
- "Liked" is a proxy for real likes, so if you have actual liked-songs data
  (Spotify's account data export includes `YourLibrary.json`), use it as the label.
