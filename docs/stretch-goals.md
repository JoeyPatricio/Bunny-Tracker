# Stretch goals

Back to the [project overview](../README.md).

1. **Water and room tracking.** The dashboard shows bowl level, today's intake
   in ml with a 7-day trend, and room temperature and humidity. Emails go out
   when the bowl is low, when intake drops well below the rabbits' own
   baseline (an early GI stasis sign), or when the room gets hot (heat stress
   starts around 28C).
2. **A paper.** A preprint whose figures and tables come from scripts in this
   repo and a frozen set of artifacts, plus a released dataset that doesn't
   include video from inside the house.

Water tracking needs the Pi's GPIO, so it builds on the
[Pi deployment](../deploy/pi/README.md). The paper can start any time.

## Non-goals

- Training on the Pi. `.venv-train` stays on the training machine.
- A vision-based drinking class. The load cell covers water.
- Multiple cameras or a database.

## Water and environment tracking

**Hardware:** a 5kg load cell with an HX711 board, a heavy ceramic bowl, two
rigid plates and M4/M5 standoffs, and a BME280 I2C sensor on 30cm+ of wire so
it sits away from the Pi's heat.

**Design:**

- A separate process, `server/sensor/water_sensor.py`, that posts to the
  server's HTTP API like the agent does.
- GPIO and I2C libraries stay out of the runtime dependencies since they don't
  install on development machines.
- Its own secret, `WATER_SENSOR_SECRET`, checked with `hmac.compare_digest`
  next to `is_agent_authed()` in `app/auth.py`. A failing sensor can never
  lock out the admin login.
- Readings go to append-only monthly JSONL (`data/water/YYYY-MM.jsonl`).
  `json_store.py` rewrites the whole file on every write, which is wrong for
  2,880 readings a day.
- Calibration is 1g = 1ml, with tare and scale set once against a known
  weight.

**Steps:**

1. **Wire the sensor and log raw data for 24h.** This is the riskiest step.
   Load cell between the plates, HX711 on 3.3V and two GPIO pins. A throwaway
   `read_hx711.py` tares, calibrates and logs grams once a second to CSV.
   Check: drift under a few grams, refills show as clean jumps, drinks as
   small drops, and bumps recover within seconds. If the data is unusable,
   fall back to level-only refill reminders.
2. **Sensor service.** Sample at 1Hz, take a 30-sample rolling median, read
   the BME280 over I2C, and POST
   `{ "grams", "tempC", "humidity", "at" }` every 30s to
   `/api/water/reading` with an `X-Water-Secret` header. Runs as
   `deploy/pi/bunnycam-water.service`. Mount the BME280 at rabbit height,
   out of the sun, 30cm+ from the Pi. Check: readings flow, a wrong secret
   gets 401, and the temperature matches a household thermometer.
3. **Server route.** `app/routers/water.py`, with the prefix added to
   `GUARDED_PREFIXES` in `app/auth.py` (otherwise `AdminGuardMiddleware`
   returns 401 before the route runs). `POST /api/water/reading` appends a
   line. `GET /api/water/summary` returns current level, today's intake,
   refills, 7-day totals and today's temperature and humidity range. Intake
   is the sum of drops. A rise over 100g is a refill. Tune the constants from
   the step 1 data. Check: a synthetic day replayed through the endpoint
   gives intake within 10% and exactly one refill.
4. **Alerts.** Reuse the existing notify path, with one cooldown gate like
   `alert_gate()` in `notify.py`.
   - Low level: under `LOW_LEVEL_G` (150), at most once per 12h.
   - Low intake: last 24h under 50% of the 7-day average, after 3+ days of
     data, at most once a day.
   - Heat: warning at `TEMP_WARN_C` (27), urgent at `TEMP_DANGER_C` (29),
     each at most once per 2h. Optionally a cold warning at 10.
   - Check: with thresholds forced, exactly one email per alert and none
     during the cooldown.
5. **Dashboard panel.** `client/src/components/WaterPanel.jsx` on the Camera
   tab, admin only. Shows level as a percent of the last refill, today's
   intake, a 7-day bar chart in plain CSS (no chart library), and
   temperature and humidity. Check: numbers match the summary endpoint and
   update after a refill.

## The paper

The paper describes an open-source, edge-deployable home-monitoring system for
rabbits: a published-ethogram vocabulary, measured label reliability, nested
grouped CV, and paired comparisons against baselines. Whether a temporal head
beats static aggregation is an open question.

1. **Write the claim first.** One paragraph and a target venue, before
   fitting any models. State the setting, clip-level weak labels, targeted
   collection, class imbalance and single camera. Call the blind re-code
   intra-observer reliability unless a second person codes a sample. An
   experiment that isn't in the claim is out of scope.
2. **Run the experiments.** Finish labeling, the blind re-code and the
   holdout cut first. Then:
   - `training.crossval` for majority, logistic regression, mean/std MLP and
     GRU on shared folds. Augmentation and crop arms once their caches exist.
   - Pose only as an offline comparison with independently annotated
     keypoints. No pose claim without grouped evaluation.
   - `training.learning_curve` for performance vs. training groups, keeping
     the outer eval sets fixed.
   - `training.figures` turns the reports in `server/models/` into
     SVG/PDF figures with JSON sidecars that record source hashes.

   Repeat SDs aren't confidence intervals. Repeated predictions aren't new
   clips. Self-agreement isn't a ceiling on model performance. The fp32 vs.
   int8 check is in-sample. Only report latency for hardware that was
   actually measured. Every plotted number has to trace back to a report and
   its hash.
3. **Freeze the artifacts.** Code revision, lockfile, ethogram version, label
   hash, manifests, seeds, CV and OOF files, all reports, ONNX files and
   figure sidecars, plus a freeze date. The paper describes that snapshot
   even if the deployed model changes later. Check: frozen reports regenerate
   every figure without refitting.
4. **Write it.** Methods cover the motion gate, two-stage inference, how
   clips get labeled, how the blind sample was drawn, and what nested
   grouped CV protects against. Limitations cover one room and camera, few
   animals, one observer, small classes, targeted collection, drift over
   time, and clip labels vs. short training windows.
5. **Release.** Only artifacts reviewed for privacy. Embeddings and the
   manifest reproduce head training without footage. They don't reproduce
   backbone, crop or pose experiments. Include scripts, lockfile, commands,
   hashes and sidecars. Leave out recordings, credentials and notes. Check:
   a clean environment reproduces the head experiments and figures.
