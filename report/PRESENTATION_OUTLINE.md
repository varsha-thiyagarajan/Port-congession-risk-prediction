# Presentation outline (10-12 minutes)

Use the PNGs in `outputs/charts/`. Demo the dashboard on slides 9 and 11.

| # | Slide | Visual | Message |
|---|---|---|---|
| 1 | Title | Dashboard screenshot | Early warning for anchorage queues at LA / Long Beach |
| 2 | Problem | none | Ships wait for berths: demurrage, fuel, crowded anchorage. Ports react late. |
| 3 | Data and cleaning | Data-quality table | AIS (14 days, 191 ships) + weather; what was removed and why |
| 4 | Where ships wait | V01, V03 | Hotspot A and Hotspot B alone hold 29% of all anchor ship-hours: monitor these two spots first. |
| 5 | How long | V05 | 105 of 123 ships reach berth in about 3.5 h; 18 wait a median of 23 h (up to 114 h). Those 18 ships cause 95% of all delay hours. |
| 6 | Who gets stuck | V06 | 36% of tankers are delayed vs 10% of cargo ships (Fisher p = 0.004): tanker berth requests need earlier confirmation. |
| 7 | What drives the queue | V08, V10, V09 | Not time of day, not weather: multi-day swings. |
| 8 | Cost | V11 | At the middle assumed rate ($30,000/ship/day) the queue cost about $2.0M in 11 days; cutting it 10% avoids about $0.20M. Rates are assumptions. |
| 9 | Forecast (live demo) | Dashboard > Forecast | Day-ahead queue and level with chance of High |
| 10 | Does ML help? | V12, V13, V14 | At 3 h the naive rule wins (error 0.79 vs 1.02); at 24 h the model cuts error by 32% (3.01 to 2.04 ships). Use the model for day-ahead planning. |
| 11 | What drives the model | shap_summary.png | The queue's own recent history dominates |
| 12 | Decisions | Dashboard > Cost & decisions | Action per client at Low / Medium / High |
| 13 | Limitations and next steps | none | More months, more ports, berth capacity data, live AIS feed |

## Speaker tip
Lead with the honest ML message: "we tested it against the simplest possible rule, and it only wins at a day ahead." It makes the rest of the project more credible.
