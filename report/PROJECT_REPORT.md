# Coastal Vessel Delay and Port Congestion Risk: Los Angeles / Long Beach

*An AIS-based early-warning system for anchorage queues.*

## 1. Problem
Ships wait at anchor when berths are busy. That costs demurrage and fuel, crowds the anchorage and makes planning hard. Ports usually react after the queue has formed. This project measures the delay, finds what is (and is not) behind it, and forecasts the queue a day ahead so each client can act early.

## 2. Data
| Dataset | Use | Status |
|---|---|---|
| NOAA/USCG AIS, 2025-10-01 to 2025-10-14 (14 days), cargo and tanker ships near LA/LB | Vessel tracks, waiting time, queue | Used |
| Open-Meteo marine and wind (hourly) | Weather features and diagnostics | Used |
| IMF PortWatch | Port-level trend | Script provided, data file not supplied, **not used** |
| AISStream live feed | Live demo | Not built; dashboard replays recorded data |

## 3. Preprocessing
| Step | Rows removed | Rows remaining | % of start |
|---|---|---|---|
| Rows in filtered AIS file (cargo + tanker, LA/LB box) | 183736 | 183736 | 100.0 |
| Invalid / missing position | 0 | 183736 | 100.0 |
| Missing or impossible speed (missing, <0 or >35 kn) | 128 | 183608 | 99.93 |
| Duplicate ship + timestamp | 11 | 183597 | 99.92 |
| Position jump (implied speed > 45 kn) | 33 | 183564 | 99.91 |
| Final analysis table (ping level) | 0 | 183564 | 99.91 |

The AIS file was already filtered to the study box, so little needed removing. Zones: berth and anchorage rectangles, a ship counts as *stationary* below 1 kn.
**Target definitions.** *Queue* = ships stationary at anchorage for less than 72 h (longer stays are lay-ups). *Time to berth* = first berth ping minus first ping inside the port area. *Delayed arrival* = more than 6 h to berth.

## 4. Key findings
1. **Most ships glide in, a few get stuck.** 105 of 123 arrivals reach berth in about 3.5 h. 18 (15%) take over 6 h, a median of 23 h and up to 114 h. They cause 95% of all delay hours.
2. **Tankers are delayed more often**: 36% vs 10% for cargo ships (Fisher exact p = 0.004).
3. **Hotspots.** Hotspot A (29 ships, 18% of anchor hours), Hotspot B (23 ships, 11% of anchor hours), Hotspot C (5 ships, 4% of anchor hours).
4. **Congestion is a multi-day swing.** The daily average queue ranges from 3.4 to 10.3 ships; the queue is High (> 8) for 21% of hours. Time of day has no significant effect (Kruskal-Wallis p = 0.36).
5. **Weather is not the culprit here.** Wave height is negatively associated with the queue, the opposite of "bad weather causes delay", and it added little to forecasts.
6. **Cost.** The queue represents 1,602 ship-hours over 10.5 days. At an assumed $30,000 per ship per day that is about $2.0M; a 10% cut would avoid about $0.20M. The rate is an assumption.

Significant statistical results:

| question | p_value | effect_size |
|---|---|---|
| Queue size differs weekday vs weekend? (only 2 weekends of data) | 0.029 | 0.1623 |
| Queue size differs by berth occupancy level (same hour, diagnostic only) | 0.0 | 0.1925 |
| Queue vs berth occupancy (raw) | 0.0 | -0.4686 |
| Queue vs berth occupancy (after removing 2-week trend) | 0.0 | -0.5315 |
| Queue vs new ships at anchor (raw) | 0.0006 | 0.2153 |
| Queue vs wave height (raw) | 0.0 | -0.3383 |
| Queue vs wave height (after removing 2-week trend) | 0.0 | -0.3043 |
| Tankers delayed > 6 h more often than cargo ships? | 0.0042 | 5.2 |
| Hours to berth differ tanker vs cargo? | 0.0064 | nan |
| Hours to berth differ by approach (West vs South)? | 0.0009 | nan |

## 5. Machine learning
**Task.** Forecast the number of ships in the queue H hours ahead, and the Low / Medium / High level (Low ≤ 5, Medium 6-8, High > 8 ships; terciles of training hours).
**Models.** Naive baselines; Linear Regression (Ridge); Random Forest; LightGBM; final = equal-weight blend of the three, each predicting the *change* from the current queue. Classifiers: LightGBM, Random Forest, Logistic Regression.
**Validation.** Rolling origin: train up to day D, predict day D+1, repeat (strictly time-ordered).

| Horizon | Naive error (ships) | Model error (ships) | Model vs naive | Model R2 |
|---|---|---|---|---|
| 3 h | 0.78 | 1.02 | -30% | 0.77 |
| 6 h | 1.16 | 1.33 | -14% | 0.63 |
| 12 h | 1.72 | 1.75 | -2% | 0.38 |
| 24 h | 3.01 | 2.04 | +32% | -0.07 |

**Result in one sentence:** the queue changes slowly, so "same as now" is hard to beat for the next few hours; the model adds real value at **24 hours ahead**, cutting average error by 32% (3.01 to 2.04 ships).

High-congestion early warning, 24 h ahead:

| model | high_caught | n_high_hours | false_alarms | recall_High | precision_High | macro_F1 |
|---|---|---|---|---|---|---|
| Baseline: class of queue now | 15 | 32 | 33 | 0.47 | 0.31 | 0.36 |
| Class from blend forecast | 20 | 32 | 25 | 0.62 | 0.44 | 0.47 |
| LightGBM classifier | 17 | 32 | 11 | 0.53 | 0.61 | 0.44 |
| Random Forest classifier | 14 | 32 | 2 | 0.44 | 0.88 | 0.47 |
| Logistic Regression classifier | 19 | 32 | 8 | 0.59 | 0.7 | 0.56 |

**What the model uses.** Only the queue's own recent history and the forecast hour (SHAP: `outputs/charts/shap_summary.png`). SHAP shows mostly mean reversion: when the recent 24 h average queue is high, the model expects it to fall back toward typical levels. Berth counts, arrivals, weekday and weather were tested and did not clear the rule "keep the simplest input set unless another cuts error by more than 10%".
| Inputs | Model error at 24 h |
|---|---|
| Queue history + time of day (final) | 2.044 |
| + berth occupancy & arrivals | 2.179 |
| + weekday | 2.177 |
| + weather (wave, wind) | 1.948 |
| Everything | 2.147 |

**Honest caveats.** R2 at 24 h is -0.07: the model beats the naive rule but explains little variance; it catches the direction of swings and misses sharp peaks. Feature sets were chosen on the same validation days, so the reported error is slightly optimistic. Weak or negative results at 3-12 h are reported, not hidden.

## 6. Decisions by client
| Level | Trigger | Client | Action |
|---|---|---|---|
| Low | queue <= 5 ships | Port authority | Normal berth schedule. Use the quiet window for maintenance and dredging checks. |
| Low | queue <= 5 ships | Shipping company | Sail at planned speed. Confirm berth window as usual. |
| Low | queue <= 5 ships | Logistics planner | Plan trucks and rail on the vessel's published ETA. |
| Low | queue <= 5 ships | Marine operations | Standard pilot and tug roster. |
| Medium | queue 6-8 ships | Port authority | Re-sequence berth slots so ships already waiting get priority. Publish the day-ahead forecast. |
| Medium | queue 6-8 ships | Shipping company | Ask for a confirmed berth window before final approach. Prepare to slow-steam to save fuel. |
| Medium | queue 6-8 ships | Logistics planner | Add a buffer of several hours to container pickup times and book extra chassis. |
| Medium | queue 6-8 ships | Marine operations | Put one extra pilot boat and tug on standby. Check anchorage spacing. |
| High | queue > 8 ships | Port authority | Prioritise berths for ships that have waited longest. Coordinate arrival slots with terminals. Open overflow anchorage. |
| High | queue > 8 ships | Shipping company | Slow steam now rather than wait at anchor. Warn charterers about demurrage risk. Send early delay alerts. |
| High | queue > 8 ships | Logistics planner | Delay truck bookings, move containers to alternative days, alert customers to late cargo. |
| High | queue > 8 ships | Marine operations | Roster extra pilots and tugs. Increase anchorage monitoring and long-stay vessel checks. |

## 7. Limitations
- 14 days, one port: weekday and seasonal effects cannot be tested; results may not generalise.
- Waiting time is estimated from geofences, not port records. Berth capacity is unknown, so occupancy is a count.
- AIS gaps: 11 low-coverage hours excluded; some ships go silent for hours.
- Only 18 delayed arrivals: vessel-type and size comparisons are indicative.
- Cost depends on an assumed demurrage rate and counts ships that may be idle for other reasons.
- Not done: PortWatch analysis, live AISStream feed, hyper-parameter tuning, Isolation Forest (dropped per the scope-control list), cargo handling time.

## 8. Visuals and their "so what"
| id | visual | client | so_what |
|---|---|---|---|
| V01 | Study area and zones | Marine operations | 21% of all pings come from ships sitting still at anchorage and 57% from berths: the zones capture real port behaviour. |
| V02 | Traffic density | Port authority | Ships funnel through two approach lanes; anchorage pressure builds just outside them, so approach scheduling is where coordination pays off. |
| V03 | Congestion hotspots (DBSCAN) | Marine operations | Hotspot A and Hotspot B alone hold 29% of all anchor ship-hours: monitor these two spots first. |
| V04 | Hourly queue timeline | Port authority | The queue ranges from 2 to 12 ships and sits in High (>8) for 21% of hours: congestion is recurring, not a one-off. |
| V05 | Time from port entry to berth | Shipping company | 105 of 123 ships reach berth in about 3.5 h; 18 wait a median of 23 h (up to 114 h). Those 18 ships cause 95% of all delay hours. |
| V06 | Delay by vessel type | Shipping company | 36% of tankers are delayed vs 10% of cargo ships (Fisher p = 0.004): tanker berth requests need earlier confirmation. |
| V07 | Arrivals by hour | Marine operations | 36% of arrivals come between 00:00 and 04:00 local time: roster pilots and tugs for that window. |
| V08 | Queue by time of day | Port authority | Queue size does not depend on time of day (Kruskal-Wallis p = 0.36): a fixed daily timetable will not fix congestion. |
| V09 | Correlations | Port authority | Rougher sea went with SHORTER queues (rho -0.34, -0.30 after removing the trend), and ships at berth rise as the queue drains (rho -0.47): the opposite of 'bad weather causes delays'. With 2 weeks of data this is association only; weather added little to forecasts. |
| V10 | Daily average queue | Logistics planner | The daily average queue swings from 3.4 to 10.3 ships: plan buffers by the day, not by the hour. |
| V11 | Demurrage scenario | Shipping company | At the middle assumed rate ($30,000/ship/day) the queue cost about $2.0M in 11 days; cutting it 10% avoids about $0.20M. Rates are assumptions. |
| V12 | Model vs naive by horizon | Port authority | At 3 h the naive rule wins (error 0.79 vs 1.02); at 24 h the model cuts error by 32% (3.01 to 2.04 ships). Use the model for day-ahead planning. |
| V13 | 24-hour forecast vs actual | Logistics planner | Forecasts made a day ahead follow the swings but miss sharp peaks (RMSE 2.5 ships): treat them as a planning signal, not a promise. |
| V14 | High-congestion warning | Marine operations | Logistic Regression classifier catches 19 of 32 High hours with 8 false alarms; the naive rule catches 15 with 33 false alarms. |
