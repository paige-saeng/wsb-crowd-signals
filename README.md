# WSB Crowd Signals

In early 2021 a Reddit forum, r/wallstreetbets, helped push GameStop from about
$20 to over $400 in a few weeks. I wanted to know whether that was a one-off or
whether the crowd's chatter carries real information about stocks. So I collected
a year of the forum's posts and comments, matched them to stock prices, and
tested it.

**Short answer:** when WSB suddenly talks about a stock much more than usual, that
stock tends to move more the next day and trades more, and over the following
month it tends to give back some of its gains. But whether the crowd sounds
bullish or bearish tells you nothing about which way the stock goes next.

*This is ongoing research. Last updated November 2026.*

## What I found

| Question | Answer |
| --- | --- |
| Does a spike in WSB attention predict a bigger price move the next day? | Yes. A 1 SD spike goes with a move about 3.5% bigger than usual (t = 6.6). About half of that turned out to be earnings news; the rest holds up after controlling for earnings (2.0%, t = 3.3). |
| Does it predict more trading volume? | Yes, and this is the sturdiest result: +2.7% volume the next day (t = 10.1), +2.0% after the earnings control. |
| What happens after a hype spike? | Across 433 spikes, stocks jumped about 4% on the day, then lagged the market by 3.7% over the next month (t = -3.2). |
| Does bullish talk predict an up move? | No. The effect is essentially zero (-0.4 basis points, t = -0.1), and it stayed zero in every version I tried. |
| Are bullish spikes followed by bigger crashes? | No. Bullish and bearish spikes gave back about the same. The crowd's mood mostly follows the price: bullish spikes came after run-ups. |

## How I did it

**Getting the data.** I downloaded the Arctic Shift archive of r/wallstreetbets
and kept all 2021 posts (1.4M) and every comment in the official Daily Discussion
threads (13M). The raw files are compressed JSON, about 6.6 GB, so I streamed them
instead of loading them at once. Each post is matched to the trading day it could
first affect: anything posted after the 4pm close counts for the next day, so no
result can peek at the future. Prices came from Yahoo Finance for 304 stocks.

**Finding which stock each post is about.** Reddit doesn't tag posts with tickers,
so I wrote rules to find them: cashtags like $GME, all-caps symbols checked against
a list of real tickers, and filters for slang that looks like a ticker (DD, YOLO,
CEO). I hand-labeled 300 posts to check it: 96% precision and 97% recall. One thing
that helped: today's ticker lists miss stocks that were renamed or delisted since
2021 (WISH, DWAC, BBBY), so I rebuilt a 2021 ticker list from a GitHub repo that
saved the exchange listings every day. Those stocks were 8% of all mentions.

**Measuring attention.** For each stock and day, abnormal attention is how far the
day's mention count sits above that stock's own average for the previous 30 days.
Comparing a stock to itself matters: 100 mentions is a huge day for a quiet stock
and an ordinary one for GME.

**Testing attention.** I built one table with a row per stock per day and ran
panel regressions with stock and date fixed effects (each stock compared to its own
normal, and each day compared across stocks so market-wide moves cancel out), with
standard errors clustered by stock and by date. For the month after a hype spike I
used an event study: abnormal returns relative to the market, added up over the
days after each spike.

**Measuring sentiment.** This was the hardest part. Standard tools read WSB badly:
VADER scored 0.29 macro-F1 on my labels and my own slang dictionary 0.39. Two
models trained on finance tweets and StockTwits posts scored 0.26 and 0.47. So I
hand-labeled about 1,000 posts myself, 300 kept aside as a test set, and trained a
logistic regression on the rest. It reached about 0.51 on the test set, better than
anything off the shelf, and I used it to score 2.7 million posts. Labeling taught me
how WSB actually talks: selling a put is bullish, anger at short sellers is bullish,
and a lot of what sounds negative is just venting about yesterday.

**Checking my methods.** Before trusting any result on real data, I ran each method
on simulated markets where I knew the right answer and made sure it found the
effect when one existed and nothing when one didn't. Those checks live in `tests/`.

## Things that went wrong

- My first ticker rules read "DON'T" as the ticker DON and found SPY inside email
  addresses. Fixed, and both cases are now tests.
- Filtering Daily Discussion threads by title caught about 170 user posts that only
  looked like official threads. I switched to the thread's flair and author.
- 74 stocks first came back with no prices. Some were rate limits, others were
  renamed symbols. Retries and a rename list brought the gap down to 54 stocks (4%
  of mentions), mostly companies that have since gone bankrupt or been bought.
- My first event study showed a -8.9% drop after hype spikes. That number was
  inflated: hyped stocks had rallied in late 2020, and the model subtracted that
  rally from every later day as if it were normal. Removing that term gave -3.7%.
- The attention effect on price shrank by about half once I controlled for
  earnings. WSB talks more about stocks right before they report.
- On my Mac, running PyTorch and LightGBM in the same Python process crashed it, so
  the embedding step runs on its own.

## Limits

- One year (2021), and January 2021, the height of the GameStop squeeze, is not in
  the tests yet because the attention baseline needs late-2020 data.
- Delisted stocks are missing prices, and many of them collapsed after their hype,
  so the true post-hype drop may be larger.
- These are associations, not proof that WSB causes the moves. Earnings are
  controlled for; other news is not.
- The sentiment model is right about half the time on individual posts. Averaging
  many posts per day cancels much of that noise, and the null result held in every
  version.
- A strategy that shorts hyped stocks would face high borrowing costs and squeeze
  risk, so the post-hype lag is not proof that anyone could have profited from it.

## What's next

- Test whether price moves come before WSB attention or after it.
- Backtest a strategy that bets against hype, including trading and borrowing
  costs.
- Add late-2020 data so January 2021 enters every test.
- Look for bot-like accounts and whether their hype behaves differently.

## Repo

```
src/data/       loading Reddit, prices, earnings dates, the trading calendar
src/nlp/        ticker extraction, sentiment scoring and models
src/features/   the stock-day table, daily sentiment
src/analysis/   regressions, event study, lead-lag tests
tests/          unit tests and the simulated-market checks
docs/           PIPELINE.md: every command to rebuild the results
```

To run it yourself, see [docs/PIPELINE.md](docs/PIPELINE.md).
