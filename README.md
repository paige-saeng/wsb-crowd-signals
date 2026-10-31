# WSB Crowd Signals

Does retail crowd attention and sentiment on r/wallstreetbets predict a stock's
next-day abnormal returns, volatility, and trading volume? And is hype from
bot-like accounts followed by a sharper reversal than organic hype?

The full research design (hypotheses, methods, pitfalls, milestones) lives in
`RESEARCH_PLAN.md`.

## Results so far (r/wallstreetbets, 2021)

WSB **attention** carries information about how much a stock will move, but WSB
**sentiment** does not tell you which way.

| Hypothesis | Question | Result |
| --- | --- | --- |
| H1 | Does abnormal WSB attention today predict a bigger price move tomorrow? | **Yes.** +1 SD attention: next-day \|return\| 3.5% above typical (t = 6.7) |
| H2 | ...and more trading volume tomorrow? | **Yes.** +2.7% next-day volume (t = 10.1) |
| H3 | Does bullish WSB sentiment predict an up move tomorrow? | **No.** +1 SD sentiment: -0.4 bp (t = -0.1); effects above about 8 bp a day ruled out |
| H4 | What happens after a hype spike? | Stocks lag by **3.7%** over the next month (t = -3.2, 433 spikes) |

All panel results use two-way fixed effects (stock and date) with standard errors
clustered by stock and by date; every method is checked on simulated data with a
known answer (`tests/`).

**Data and measurement**
- 1.4M posts and 13M Daily Discussion comments (Arctic Shift dumps), each mapped to
  the trading day it could affect (no lookahead).
- Rule-based ticker extraction: 3.4M mentions, precision 0.96 and recall 0.97 on 300
  hand-labeled texts. A point-in-time 2021 ticker list recovers renamed and delisted
  names (8.1% of mentions) that today's listings miss.
- Sentiment: off-the-shelf scorers read WSB poorly (VADER macro-F1 0.29, a custom WSB
  lexicon 0.39, finance-tuned transformers 0.26 to 0.47). A model trained on 755
  hand-labeled WSB texts reaches 0.51 to 0.55 on a separate 300-text test set and
  scored 2.7M texts.
- Prices: Yahoo Finance for 304 stocks, with symbol-rename recovery; 54 delisted
  names (4% of study mentions) remain missing.

Full findings log, decisions and caveats: the project's research doc (`RESEARCH_PLAN.md`
holds the original design).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest     # unit tests and known-answer simulations, offline
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

**Ticker universe (point in time).** `data/reference/tickers_2021.csv` lists every
NASDAQ/NYSE/AMEX symbol seen in daily snapshots from 2021-01-30 to 2022-01-29
(9,077 tickers), so 2021 names later renamed or delisted (WISH, DWAC, CCIV, BBBY)
are still recognized. Rebuild it with:

```bash
python -m src.data.universe --start 2021-01-01 --end 2022-01-31 --out data/reference/tickers_2021.csv
```

Optionally also download today's NASDAQ Trader lists into `data/raw/` and pass
them with `--symbols`:
[nasdaqlisted.txt](https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt),
[otherlisted.txt](https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt).

**Run the extractor** on the Phase 1 output:

```bash
python -m src.nlp.tickers \
  --inputs data/interim/posts.parquet data/interim/comments.parquet \
  --universe-csv data/reference/tickers_2021.csv \
  --symbols data/raw/nasdaqlisted.txt data/raw/otherlisted.txt \
  --out data/processed
```

Writes `mentions_raw.parquet` (before filtering), `mentions.parquet`,
`daily_mentions.parquet` (mentions and unique authors per ticker per trading
day), and `ambiguous_ticker_decisions.csv`. To redo only the filter step:
`python -m src.nlp.tickers --skip-extract --out data/processed`.

**How matching works:** cashtags (`$GME`) always count; bare ALL-CAPS symbols
count unless on a stoplist of slang (DD, YOLO, CEO…); a short list of distinctive
symbols (gme, amc, pltr…) also counts in lowercase. Links, emails, r/ and u/
references are ignored; contractions (DON'T) don't match. Each ticker counts once
per post or comment. Then the **ambiguity filter**: tickers that are common
English words or two letters keep their non-cashtag mentions only if at least 5%
of their mentions are cashtags (and 20+ cashtags). Joke cashtags ($ROPE) are
excluded.

**Validation:** 300 hand-labeled texts in `data/labels/label_sample.xlsx`.
Score the current pipeline against them:

```python
from src.nlp.tickers import load_labels, repredict, score_labels, load_cashtag_only, build_universe
from src.data.universe import load_universe_csv
u = build_universe(load_universe_csv("data/reference/tickers_2021.csv"))
co = load_cashtag_only("data/processed/ambiguous_ticker_decisions.csv")
score_labels(repredict(load_labels("data/labels/label_sample.xlsx"), u, co))
```

## Phase 3-4: does attention predict volatility and volume? (H1, H2)

1. Download prices for the 361 study tickers (500+ mentions on 30+ days in 2021),
   plus SPY, with a year of history before 2021 for baselines:

```bash
python -m src.data.prices --ticker-file data/processed/study_tickers.txt \
  --start 2020-06-01 --end 2022-02-01 --out data/raw/prices.parquet
```

2. Build the stock-day panel (features on day t, outcomes on day t+1):

```bash
python -m src.features.panel
```

3. Run the regressions (two-way fixed effects, errors clustered by stock and date):

```bash
python -m src.analysis.attention
```

Results print to the terminal and save to `reports/tables/attention_results.csv`.
Each hypothesis runs on the full sample, without GME and AMC, without the
Jan-Feb squeeze, and on each half of 2021, plus alternative attention measures.
`tests/test_panel_attention.py` checks the method on simulated markets with a
known answer: it must find a planted effect and must not find one that isn't there.

## Phase 4: event study of hype spikes (H4)

What happens to a stock's price before, on, and after a WSB hype spike?

```bash
python -m src.analysis.event_study
```

- **Event:** a stock-day with attention at least 3 standard deviations above the
  stock's own 30-day norm and at least 10 mentions. Spikes within 20 trading
  days of the stock's previous spike belong to the same episode.
- **Abnormal return:** beta-only model (stock return minus beta x SPY return),
  beta estimated on days -250 to -30 before each event, alpha set to 0. WSB
  stocks' late-2020 rally gives them a large estimated alpha (+0.26%/day), which
  the full market model would subtract every day and turn into a fake ~5%
  decline over 20 days. Market-adjusted and full-market-model results are
  reported as variants.
- **Windows:** before (-10,-1), day 0, next day, next week (+1,+5), next month
  (+1,+20). Days +1 onward are what a trader could act on after the spike.
- **Inference:** winsorized means with standard errors clustered by event date
  (many stocks spike on the same day), plus medians and share positive.
- **Groups:** all events, price up vs down on day 0, without GME and AMC,
  without the Jan-Feb squeeze.

Outputs: `reports/tables/event_study_car.csv` (windows x groups),
`reports/tables/event_study_path.csv` (average cumulative abnormal return by
day, with 95% bands), `data/processed/event_returns.parquet` (per event and day).
`tests/test_event_study.py` plants a known jump-then-reversal in simulated
stocks and checks it is recovered, and that no reversal is found when none exists.

## Sentiment model and H3: does WSB tone predict direction?

1. Hand labels: `data/labels/sentiment_sample.xlsx` (300 test texts) and
   `data/labels/sentiment_train.numbers` (training texts, drawn with
   `python -m src.nlp.sentiment sample`, never overlapping the test set).
2. Compare scorers and train models (graded only on the test labels):

```bash
python -m src.nlp.sentiment evaluate                 # VADER and WSB lexicon
python -m src.nlp.sentiment_model train              # TF-IDF logistic regression and LightGBM
python -m src.nlp.sentiment_model explain            # words the model relies on
python -m src.nlp.sentiment_model embed              # optional: sentence embeddings (PyTorch)
python -m src.nlp.sentiment_model train --embed
python -m src.nlp.sentiment_model finance-models     # optional: finance-tuned transformers, zero-shot
```

On macOS, PyTorch and LightGBM in one Python process can crash, so the embedding
and transformer steps run as their own commands.

3. Score every text, average per stock per day, and test H3:

```bash
python -m src.features.sentiment_daily score         # ~2.7M texts, resumable
python -m src.features.sentiment_daily daily
python -m src.analysis.direction                     # H3 regressions
python -m src.analysis.hype_sentiment                # H3 x H4: tone of hype spikes
```

`sent_net` = mean of P(bullish) - P(bearish) over a stock's texts that day; days
with fewer than 5 texts are left out.

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
