# ANCHOR data card

Generated 2026-11-23T20:35:20Z from `markets.csv` (SHA-256 `480bb8d3172dbe8c...`).

## Sources and pull

- **polymarket**: pulled 2026-11-23T18:46:32Z; endpoints: `https://gamma-api.polymarket.com/events`, `https://clob.polymarket.com/prices-history`; 1618 API calls.
  - Top 10 pages (100 per page) of closed events by volume in each 30-day window of end dates: 23587 events; 1380 had an eligible market; seeded random sample of 1380 events.
- **kalshi**: pulled 2026-11-23T18:55:55Z; endpoints: `https://api.elections.kalshi.com/trade-api/v2/events`, `https://api.elections.kalshi.com/trade-api/v2/historical/markets`, `https://api.elections.kalshi.com/trade-api/v2/series`, `https://api.elections.kalshi.com/trade-api/v2/series/{series}/markets/{ticker}/candlesticks`, `https://api.elections.kalshi.com/trade-api/v2/historical/markets/{ticker}/candlesticks`; 17132 API calls.
  - Series-first sampling: 6000 of 16239 Kalshi series (hourly/daily series excluded) scanned in seeded random order; at most 5 events with an eligible market taken per series, 898 events in total. Historical cutoff: `{'market_positions_last_updated_ts': '2026-09-24T00:00:00Z', 'market_settled_ts': '2026-09-24T00:00:00Z', 'orders_updated_ts': '2026-11-09T00:00:00Z', 'trades_created_ts': '2026-09-24T00:00:00Z'}`.

## Eligibility rules

- Binary Yes/No market that resolved cleanly to YES or NO.
- Its event is a single Yes/No question, or exactly two mutually exclusive options (for example Team A wins or Team B wins). Events with three or more options (candidate lists, price ladders, score brackets) are excluded.
- Created on or after 2025-01-01 (earliest of the platform's creation and start timestamps).
- Resolved (trading closed) before the pull time.
- Snapshot price taken 7 days before resolution: the last price point at most 48 hours before that moment. Polymarket: CLOB `/prices-history` price of the Yes token. Kalshi: hourly candle bid/ask midpoint when the spread is at most 0.2, otherwise last trade.
- Snapshot price between 0.05 and 0.95.
- Not a price-threshold question ("Bitcoin above $X on date").
- Not a vague question ("Nothing Ever Happens: ...") and no relative date ("by Friday", "this week").
- Not on the hand-review exclusion list (`manual/exclude_markets.csv`: questions that do not state what is being predicted, or use relative dates such as "by Friday").
- Volume at least 10000 USD (Polymarket) or 10000 contracts (Kalshi).
- One market per event (random choice, seed 3346).
- Cross-platform duplicates (same question, resolution within 3 days) removed, keeping the higher-volume copy. Auto threshold 0.9; pairs from 0.6 checked by hand.
- At most 15 markets from any one Kalshi series (all soccer 'both teams score' series count as one, and all US election races count as one), and at most 3 versions of the same question with different deadlines.
- Platforms kept as even as possible, each at most 55% of the final dataset.
- Category share caps on the final dataset: politics at most 25%.
- Downsampled to at most 300 markets, stratified by platform and category.

## Markets dropped by each filter

| step | kalshi | polymarket |
|---|---|---|
| markets retrieved | 159072 | 504124 |
| dropped: not_binary_yes_no | 4260 | 307334 |
| dropped: event_has_3plus_options | 149892 | 195223 |
| dropped: no_clean_resolution | 0 | 2 |
| dropped: missing_dates | 0 | 0 |
| dropped: created_before_cutoff | 266 | 3 |
| dropped: resolved_after_pull | 0 | 0 |
| dropped: open_too_short_for_snapshot | 3540 | 393 |
| dropped: below_volume_threshold | 521 | 6 |
| dropped: not_sampled | 0 | 0 |
| dropped: no_price_at_snapshot | 65 | 16 |
| dropped: snapshot_price_out_of_range | 205 | 582 |
| dropped: price_threshold_question | 1 | 160 |
| dropped: vague_or_relative_date_question | 10 | 41 |
| dropped: manually_excluded | 1 | 3 |
| dropped: one_market_per_event | 70 | 0 |
| dropped: cross_platform_duplicate | 2 | 0 |
| dropped: topic_cap | 15 | 23 |
| dropped: category_cap | 72 | 173 |
| dropped: downsampled_to_target | 2 | 15 |
| final dataset | 150 | 150 |

## Final counts by category and platform

| category | kalshi | polymarket | total |
|---|---|---|---|
| crypto | 1 | 34 | 35 |
| culture | 31 | 16 | 47 |
| economics | 15 | 34 | 49 |
| other | 1 | 1 | 2 |
| politics | 38 | 37 | 75 |
| science_tech | 2 | 10 | 12 |
| sports | 60 | 14 | 74 |
| weather | 2 | 4 | 6 |
| total | 150 | 150 | 300 |

## Summary statistics

- Markets: 300. YES rate: 0.547.
- Snapshot price: mean 0.439, median 0.405.
- Brier score of the snapshot price itself (parrot baseline, informed condition): 0.1788.
- Resolution dates: 2025-01-23 to 2026-11-23.

## Dev/test split (seed 3346, stratified by category)

| category | dev | test | total |
|---|---|---|---|
| crypto | 10 | 25 | 35 |
| culture | 14 | 33 | 47 |
| economics | 15 | 34 | 49 |
| other | 1 | 1 | 2 |
| politics | 22 | 53 | 75 |
| science_tech | 4 | 8 | 12 |
| sports | 22 | 52 | 74 |
| weather | 2 | 4 | 6 |
| total | 90 | 210 | 300 |

## Known limitations

- Category labels come from keyword matching on platform tags (Polymarket) and series categories (Kalshi).
- Volume units differ: USD on Polymarket, contracts on Kalshi.
- Polymarket's resolution time is the market's closed time, which can lag the end of trading.
- Within-platform near-duplicates in different events are not removed.
