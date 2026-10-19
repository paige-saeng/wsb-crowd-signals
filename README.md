# WSB Crowd Signals

Does retail crowd attention and sentiment on r/wallstreetbets predict a stock's
next-day abnormal returns, volatility, and trading volume? And is hype from
bot-like accounts followed by a sharper reversal than organic hype?

The full research design (hypotheses, methods, pitfalls, milestones) lives in
`RESEARCH_PLAN.md`.

> Status: **Phase 1, setup and data.** Findings will go here as they come in.

## Setup

```bash
conda env create -f environment.yml
conda activate wsb-crowd-signals
pytest            # 30 tests, runs offline in a few seconds
```

## Phase 1: getting the data

### 1. Reddit (r/wallstreetbets)

**Recommended: Arctic Shift subreddit dumps.** Full history of the subreddit,
posts *and* comments, with authors, which the Daily Discussion and bot-scoring
work needs. Download links are in the Arctic Shift GitHub repo
(`download_links.md`); the files are torrents on Academic Torrents. Grab the
per-subreddit files for wallstreetbets only, not the whole-Reddit monthly dumps:

```
data/raw/wallstreetbets_submissions.zst
data/raw/wallstreetbets_comments.zst
```

The comments file is very large. Process a window, e.g. 2020 through 2022:

```bash
python -m src.data.reddit \
  --submissions data/raw/wallstreetbets_submissions.zst \
  --comments    data/raw/wallstreetbets_comments.zst \
  --start 2020-01-01 --end 2023-01-01 --out data/interim
```

Add `--daily-only` to keep only Daily Discussion / What Are Your Moves Tomorrow /
Weekend thread comments; much smaller, and a good first run.

| Option | Posts | Comments | Authors | Range | Use for |
| --- | --- | --- | --- | --- | --- |
| Arctic Shift subreddit dumps | Yes | Yes | Yes | Full history | The main dataset |
| Kaggle "Reddit WallStreetBets Posts" | Yes | No | No | Early 2021 | Quick prototyping only (`load_kaggle_csv`) |
| SocialGrep WSB Aug 2021 (Hugging Face) | Yes | Yes | Yes | One month | Testing code on a small real sample |

### 2. Prices, volume, and the market benchmark

```bash
python -m src.data.prices --tickers GME AMC BB NOK BBBY TSLA PLTR \
  --start 2019-01-01 --end 2023-03-01 --out data/raw/prices.parquet
```

SPY is always added. Start about a year before the Reddit window so beta can be
estimated before the earliest events. The script prints a coverage report;
tickers with no data (usually delisted ones) are your survivorship-bias gap.
In Phase 2 the ticker list will come from the tickers WSB actually mentions.

## Phase 2: ticker extraction

1. Download the free NASDAQ Trader symbol lists into `data/raw/`:
   [nasdaqlisted.txt](https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt) and
   [otherlisted.txt](https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt).
   They list only stocks listed *today*, so add delisted meme names with `--extra`.
2. Run the extractor on the Phase 1 output:

```bash
python -m src.nlp.tickers \
  --inputs data/interim/posts.parquet data/interim/comments.parquet \
  --symbols data/raw/nasdaqlisted.txt data/raw/otherlisted.txt \
  --extra BBBY --out data/processed
```

Writes `data/processed/mentions.parquet` (one row per post/comment and ticker)
and `data/processed/daily_mentions.parquet` (mentions and unique authors per
ticker per trading day), and prints the top 20 tickers.

**How matching works:** cashtags (`$GME`) always count; bare ALL-CAPS symbols
count unless they're on a stoplist of words and slang (DD, YOLO, CEO, ALL, IT…);
a short list of distinctive symbols (gme, amc, pltr…) also counts in lowercase.
Links, emails, and r/ or u/ references are ignored. Each ticker counts once per
post or comment.

**Validate it (required before trusting results):** draw a sample, hand-label
it, and measure precision and recall.

```python
from src.nlp.tickers import sample_for_labeling, score_labels
sample_for_labeling(comments, universe, n=300).to_csv("data/interim/label_sample.csv", index=False)
# fill the true_tickers column in Excel/Sheets, then:
score_labels(pd.read_csv("data/interim/label_sample_labeled.csv"))
```

## The timestamp rule (lookahead bias)

Every post and comment gets a `trading_day`: **the first trading session whose
close is strictly after the post.** A post at 3:59 pm ET counts for that day; at
4:00 pm or later it counts for the next trading day; weekends and holidays roll
forward; early-close days use the real 1:00 pm close; daylight saving is handled
by the NYSE calendar. So features for day *d* only use what was posted before
day *d*'s close, and targets are measured from day *d+1*. See
`src/data/calendar.py` and `tests/test_calendar.py`.

## Repo layout

```
src/data/       loaders: reddit dumps, prices, trading calendar   (Phase 1)
src/nlp/        ticker extraction, sentiment, bot scoring          (Phase 2)
src/features/   attention, sentiment, controls                     (Phase 3)
src/finance/    returns, beta, abnormal returns, CAR               (Phase 3-4)
src/analysis/   event study, regressions, Granger                  (Phase 4)
src/backtest/   strategy rules, costs, metrics                     (Phase 5)
src/live/       paper trading job                                  (Phase 7)
notebooks/      the story, calling code in src/
tests/          unit tests
data/           raw / interim / processed (gitignored)
```
