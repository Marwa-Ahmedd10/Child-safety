# Human Detection

Live person detection, current visible-person counting, child/adult labelling, safety alerts and visitor counts, using RF-DETR, ByteTrack, DINOv2 and OpenCV. Supports a macOS webcam, video files, or one RTSP camera. Detection, tracking and the child labels use only models and libraries licensed for commercial use (see [Licences](#licences)).

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) from `pyproject.toml`, with exact versions pinned in `uv.lock`. uv installs Python 3.12 if needed.

```sh
uv sync
uv run download-models
uv run live-camera --child
```

Add a dependency with `uv add <package>` and upgrade with `uv lock --upgrade-package <package>`; commit `uv.lock` so every machine installs the same versions.

Allow camera access for your terminal in macOS System Settings → Privacy & Security → Camera. Press Q or Escape to quit. `download-models` fetches the person detector (about 370 MB; add `--detector small` for another size), the child classifier's image encoder and the age model; inference then runs locally without saving video.

Count only: `uv run live-camera`. Alternative camera: `--source 1`. RTSP or another stream URL: `--source 'rtsp://CAMERA_HOST:554/STREAM_PATH'`. Video: `--source /path/to/video.mp4`. A webcam or network camera is read on its own thread and only its newest frame is processed, so the app never falls behind the camera. The device is picked automatically (`--device auto`: CUDA, then Apple Silicon `mps`, then CPU); force one with `--device cpu` or `--device mps`.

Run the checks with `uv run python -m unittest` and the linter with `uv run ruff check .`; `.github/workflows/ci.yml` runs both on every push. The tests use stand-ins for the models, so they need no weights.

## Project layout

The code is one package, `src/human_detection`, installed by `uv sync` with these commands (run each with `uv run`; `--help` lists its options):

| Command | What it does |
|---|---|
| `live-camera` | Counts, labels and checks one camera, video or webcam against its zones |
| `run-site` | Runs every camera of a site, one `live-camera` process each |
| `download-models` | Downloads the model weights, so the app can run without internet |
| `draw-zones` | Draws zones and counting lines on a camera's view |
| `footage` | Turns your own footage into labelled pictures, and measures the child classifier on them |
| `build-child-dataset`, `train-child-classifier` | Rebuild the training data and retrain the child classifier |

```
src/human_detection/
  camera.py          frames from webcams, video files and stream URLs, reconnecting when a stream drops
  detection/         the RF-DETR person detector (detector.py) and tracking between frames (tracking.py)
  labels/            whole-body child labels (child.py), face-based age (face_age.py), settling per person (votes.py)
  safety/            zones (zones.py), the alert rules (monitor.py) and where alerts go (events.py)
  visitors.py        visitor counts at counting lines
  app/               live-camera: options (cli.py), the frame loop (pipeline.py), camera health (health.py),
                     the preview window (overlay.py)
  site.py            run-site
  downloads.py       download-models
  zone_editor.py     draw-zones
  training/          the child classifier's data (openimages.py, footage.py, data.py), head (head.py),
                     scores (metrics.py) and training run (train.py)
tests/               unit tests, one file per area
models/              downloaded weights, and the child classifier head (child_head.npz, kept in Git)
```

## Person detector

People are found by [RF-DETR](https://github.com/roboflow/rf-detr) and followed between frames by ByteTrack from [trackers](https://github.com/roboflow/trackers); both are Apache-2.0. At start-up the model is traced, and runs in half precision on a GPU (Apple or NVIDIA). Choose a size with `--model`, and download it first with `uv run download-models --detector SIZE`:

| `--model` | COCO accuracy (AP) | NVIDIA T4 (published) | M1 MacBook Air, 8 GB (measured) | Download |
|---|---|---|---|---|
| `nano` (default) | 48.4 | 2.3 ms | 70 ms (14 FPS) | 366 MB |
| `small` | 53.0 | 3.5 ms | 113 ms (9 FPS) | 386 MB |
| `medium` | 54.7 | 4.4 ms | not measured | 405 MB |
| `large` | 56.5 | 6.8 ms | not measured | 136 MB |

YOLO11n, which this replaced because its licence (AGPL-3.0) does not allow closed commercial use, scores 39.5 AP. On a crowded mall clip filmed from an upper floor, Nano found 14 people in a frame where YOLO11n found 6, all of them real, and Small found 15. Neither found the small, distant people packed on the escalators: frames are shrunk to the model's input size (384 px for Nano), and running Nano at a larger size did not help. For wide, high-resolution views, compare `medium` or `large` on footage from the real cameras.

On the M1, the child labels add about as much time per frame as the detector in a crowd, because each person's crop is read every third frame.

When the tracker loses someone and, within 2 seconds, starts a new track where they were last seen with nobody else near that spot, the person keeps their old ID, so they are not counted twice. Their child label starts again at Checking, so a child can never inherit an adult's label. In crowds this is ambiguous, so people there get a new ID as before; on a crowded mall clip at 7.5 FPS, the rule reduced the extra new IDs from 15 to 6. Someone who leaves the view for longer comes back as a new person, because nothing recognises faces or identities.

## Counting

The count covers people currently visible, not all room occupants or cumulative visitors. Hidden people may be missed. Each person gets a track ID (`#3`) that follows them between frames. A person tracked for at least `--hold` frames (default 10) stays counted for up to that many missed frames, drawn with a thin box, so a single missed detection does not change the count; brief false detections are not held. Use `--hold 0` to count only the current frame.

## Child detection (`--child`)

What the classifier is for, how it was trained and measured, and where it fails are summarised in [MODEL_CARD.md](MODEL_CARD.md); every alert records the model versions it was made with.

Labels each tracked person **Child**, **Adult** or **Checking**, from their whole body, so it works when faces are small, turned away or seen from above. A person is Checking until they have three readings; **hazard and exit alerts treat Checking as a possible child**. Each label is the average of that person's last 10 readings, taken about every third frame. People less than 64 px tall in the image, such as a head at the edge of the frame or someone far away, are too small to judge: they get no reading, so they keep the label from earlier, larger views, or stay Checking.

In the preview, children are boxed in yellow and adults in blue, each with a tag such as `CHILD 0.87  #3`: the label, a score and the track ID. The score is the average P(child) behind a Child label, or 1 minus it behind an Adult one; it shows how clear-cut the label is, not a calibrated probability. People still Checking get a thin grey box without a tag.

The classifier is a frozen DINOv2-small image encoder plus a small logistic-regression head (`models/child_head.npz`), trained on photo crops and on copies degraded to look like webcam or CCTV images (small, blurred, noisy, badly lit, compressed, sometimes cut off below the chest).

`--child-recall` sets the trade-off: the share of children it should find. Higher misses fewer children but flags more adults. The default, 0.95, puts safety first; in a space with few children it will raise false alarms. Measured on 514 held-out people (257 children, 257 adults) from Open Images photos, as children found / adults flagged:

| `--child-recall` | Photos | People 64 px tall | Webcam-quality copies |
|---|---|---|---|
| 0.95 (default) | 96% / 19% | 96% / 30% | 92% / 32% |
| 0.90 | 93% / 11% | 93% / 18% | 88% / 18% |
| 0.85 | 88% / 7% | 89% / 11% | 80% / 11% |
| 0.80 | 83% / 4% | 84% / 5% | 71% / 7% |

These are single readings; each person's label averages up to 10 readings over time. On the same photos, face-based age (`--age`) found 30% of children, flagged 7% of adults, and could not judge 65% of people (no usable face).

Limits of these numbers:
- The test photos are mostly eye-level, often close-ups; ceiling CCTV views are not represented. **Measure on footage from the real cameras before relying on it.**
- About 5–10% of the dataset labels look wrong (mostly teenagers labelled as adults), so measured accuracy is somewhat understated.
- Teenagers are inherently ambiguous; "child" here means visibly a child.
- These are appearance estimates, not verified ages. Use them to alert staff, not as the only safeguard.

Retrain or re-measure (writes `models/child_head.npz` and `models/child_head_report.json`; downloads about 650 MB):

```sh
uv run build-child-dataset
uv run train-child-classifier
```

### Footage from your own cameras

**From above, the classifier labels most adults as children.** On a mall clip filmed from an upper floor, it flagged 85% of adults at the default `--child-recall 0.95`, 16% at 0.85 and 2% at 0.80. How many children each setting still finds from above can only be measured on labelled footage from cameras like ours, which is also what it needs to learn. That footage does not need a hazard, or the site itself: the classifier judges how a person looks from the camera's height and angle, not where they are, so any room seen by a camera mounted like the product's will do. Record it following the team's recording guide, then:

```sh
uv run footage extract lobby_2026-11-03_cam1_s1.mp4   # pictures of each person, per session
uv run footage label data/site_crops/lobby_2026-11-03_cam1_s1   # C child, A adult, S skip
uv run footage evaluate                                # how the current classifier does
uv run train-child-classifier --site data/site_crops --test-session lobby_2026-11-05_cam1_s3
```

- `extract` follows each person with the app's detector and tracker and saves up to 10 pictures of them, 1 second apart. In crowds the tracker sometimes passes an ID from one person to another, so a track's pictures are split wherever the person's appearance suddenly changes; on the mall clip that split 81 tracks into 98 people. Splitting misses subtle swaps, so press S for any row still showing more than one person.
- `label` runs on this computer, and the pictures never leave it. Labels are saved after every key.
- `evaluate` averages each person's pictures, as the app does, and reports children found and adults flagged at each `--child-recall`.
- `--site` adds the labelled sessions to training, keeping all of one person's pictures in the same cross-validation fold. `--test-session` holds one session back and reports per-person results on it: that is the honest measure from above. Once the training footage holds at least 10 children, the `--child-recall` thresholds are set from them instead of from the photos. Use `--out` to save somewhere other than `models/child_head.npz`.
- `data/` is excluded from Git; keep the footage and pictures in the project's encrypted folder, as the guide says.

## Safety rules (`--zones`)

Draw zones on a camera's view, then run with them (this turns on `--child` when a zone needs child labels):

```sh
uv run draw-zones --source 0 --out zones.json
uv run live-camera --zones zones.json --camera-name "Lobby" --events alerts.jsonl --webhook https://your-system/alerts
```

To try it on a webcam, use `zones.example.json`: the left 30% of the view is a hazard zone, the right 25% an exit, the whole view a "Room" allowing one child and one child per adult, and a counting line down the middle ("in" is walking to the right).

| Zone type | Alert |
|---|---|
| `hazard` | `child_in_hazard_zone`: one or more possible children stand in the zone. With `"unaccompanied": true`, `child_alone_in_hazard_zone`: a possible child stands in it with no confirmed adult within `accompany_distance`. With `"anyone": true`, `person_in_hazard_zone`: anyone stands in it, whatever their label |
| `exit` | `child_exiting_alone`: one or more possible children stand in the zone with no confirmed adult within `accompany_distance` body heights (default 1.5) |
| `count` | `too_many_children`: more than `max_children`; `too_few_adults`: more than `max_children_per_adult` children per adult (with no adults, any child) |
| `ignore` | People standing here are dropped before counting: TVs, posters, mirrors, glass |
| `line` | Not an alert: a counting line for visitor counts (see below) |

### Hazard areas

A hazard zone is any area children should not be in, or not be in alone: nothing in the rules is specific to one kind of hazard. These options fit a zone to its hazard:

| Option | Effect |
|---|---|
| `"anyone": true` | Alert on anyone, whatever their label, for places nobody should be at certain times. It does not use the child label, so it cannot miss an older child |
| `"unaccompanied": true` | Alert only on a possible child with no confirmed adult within `accompany_distance` |
| `"after_seconds": 5` | Alert only once one person has stayed in the zone that long, so someone walking past does not raise it. Leaving the zone restarts the wait, and so does a new track ID. Hazard and exit zones |
| `"alert": "Child near the stairs"` | The alert's text, in the banner and the event's `message`. Hazard and exit zones |
| `"hours": ["20:00-07:00"]` | Only at these local times. Any zone |

Starting points; check each on the camera before relying on it:

| Hazard | Zones |
|---|---|
| Road, car park, driveway, loading bay | A hazard zone on the area itself; an exit zone on the gate or door that leads to it |
| Stairs, escalator, lift | `"unaccompanied"` hazard zones on the landings at the top and the bottom |
| Water: pool, pond, fountain | A hazard zone on the floor along the edge; an `"anyone"` zone with the hours it is closed |
| Kitchen, plant room, machinery, storage, staff-only room | A hazard zone, or `"unaccompanied"` where staff may bring a child in; an `"anyone"` zone outside working hours |
| Balcony, roof, open window | A hazard zone on the floor in front of the railing or window |

Use `after_seconds` only where people often cross a zone without being at risk, such as a walkway past a doorway: it delays every alert from that zone by that long.

A zone can be tried out without the hazard: draw it over any floor area, mark the same area on the floor with tape, and walk through the cases it must get right: a child alone, a child with an adult, a child passing by, an adult alone.

How the rules decide:
- In hazard and exit zones, a possible child is anyone labelled Child or still Checking. Count zones count only people labelled Child or Adult: someone still Checking counts as neither, so a count alert can start about a second later. Only a confirmed Adult accompanies a child.
- A person's position is the bottom centre of their box, where they stand, so draw zones on the floor. Someone standing in front of a TV is not ignored; a person shown on it is.
- The child label can miss children, especially older children (10–12) and children in swimwear: on a clip of a boy at a pool edge it scored him as an adult at every setting. Where nobody should be at certain times, use an `"anyone": true` hazard zone, which does not depend on the label at all, and set its `"hours"`.
- Any zone can apply only at set local times: `"hours": ["20:00-07:00"]` (a span ending before it starts runs overnight; several spans allowed). Outside them it raises nothing.
- A hazard or exit alert starts once one person has met its condition for `confirm_frames` frames in a row (default 10, about a second), so the moment of Checking that every new track ID goes through cannot add up to a false alert; a count-zone alert needs its condition for `confirm_frames` frames in a row. An alert ends after `release_frames` frames (default 30) with nobody meeting it, or, when the app stops, with an end event marked `"reason": "monitor stopped"`. Each alert produces a start and an end event; the window shows active alerts in a red banner.
- Each zone raises at most one alert of each kind at a time, whoever is in it. Once on, it stays on while anyone meets its condition, so a child whose track ID changes neither ends it nor raises a new one. The banner shows the current number of children; a second child entering a zone that is already alerting does not send a new event.
- Distances are measured in the image relative to body height, a rough stand-in for metres until cameras are calibrated.
- Draw hazard and exit zones on the floor in front of the hazard, where a person would stand, and where people appear at least 64 px tall. Smaller people, such as people far from the camera or swimmers with only their heads above water, are never judged and stay Checking, which counts as a possible child, so a zone over them alerts on everyone. People partly hidden, in water or behind counters, railings or parked cars, are also judged less well: with only their upper body showing, adults score as children. For such an area itself, use an `"anyone"` zone with the hours nobody should be there.

Each event is one JSON line in the `--events` file, and the same JSON is POSTed to `--webhook`. End events add `duration_s`. No images are stored or sent. A failed webhook post is retried 4 more times, waiting 1, 2, 4 and 8 seconds; the events file is the full record either way.

**Camera health.** When a live camera stops delivering frames for 5 seconds, cannot be opened, or its reader fails, a `camera_offline` alert starts ("nobody is being monitored") and the app keeps reopening it, waiting 1, 2, 4… up to 30 seconds between tries; the alert ends when frames return, with the time since the last frame before the outage. Folders for `--events`, `--counts` and `--status` are created if missing, and a file that cannot be written is logged without stopping the monitoring; the webhook still gets the alert. `--status status.json` rewrites a small file every 5 seconds with the camera's state (online, drops, FPS, frames, people, active alerts) for a monitoring system to watch; a file that stops changing means the app itself has stopped. Stopping the app with Ctrl-C or SIGTERM (as Docker and systemd do) closes open alerts and saves counts.

```json
{"time": "2026-10-27T10:05:07+00:00", "camera": "Lobby", "event": "child_in_hazard_zone", "state": "start", "zone": "Kitchen", "children": 1, "track_ids": [1], "message": "1 child in hazard zone \"Kitchen\""}
```

Current limits: one camera per process; the tracker sometimes follows one person as two, so a count can be too high.

### Visitor counts

Draw a counting line across an entrance (two clicks, then Enter, in `draw-zones`). Each tracked person whose feet cross it, ending at least a quarter of their height past it, is counted once per crossing, so someone sitting or swaying on the line is not: "in" towards the side its arrow points to, "out" the other way; to flip it, draw its points in the other order. Each crossing is counted with the person's child/adult label, after waiting up to 30 frames for it to settle.

```sh
uv run live-camera --zones zones.json --camera-name "Lobby" --counts counts.jsonl --count-every 60
```

Every `--count-every` minutes (default 60) and on exit, each line's counts for that period are printed and appended to the `--counts` file; the window shows running totals on the line:

```json
{"time": "2026-10-27T15:00:00+00:00", "camera": "Lobby", "line": "Main entrance", "from": "2026-10-27T14:00:00+00:00", "to": "2026-10-27T15:00:00+00:00", "in": {"total": 41, "children": 9, "adults": 30, "age_unknown": 2}, "out": {"total": 37, "children": 7, "adults": 28, "age_unknown": 2}}
```

Limits: a person who turns back on the line, or whose track ID changes as they cross, can be counted twice or missed; place lines where people walk straight through, fully in view. The child label's limits above apply to every count.

## Running a site

`run-site` runs every camera of a site from one TOML file: one `live-camera` process per camera, each restarted if it stops (after 5 seconds, doubling to at most 5 minutes while it keeps failing within a minute of starting). Ctrl-C or SIGTERM stops them all cleanly. Each camera's output is prefixed with its name.

```sh
cp site.example.toml site.toml   # list the cameras, their zones and outputs
export LOBBY_RTSP='rtsp://user:pass@10.0.0.21:554/stream1' CARPARK_RTSP=... ALERT_WEBHOOK=...
uv run run-site site.toml
```

Each `[[camera]]` takes `live-camera`'s options without the dashes (`child_recall`, not `--child-recall`) plus a `name`; `[defaults]` holds options shared by every camera, where `{camera}` becomes the camera's name made safe for file names. Any `source` or `webhook` written `env:NAME` is read from the environment variable `NAME`, so camera passwords and webhook tokens stay out of files and shell history; RTSP passwords are never written to logs or alerts.

With Docker (`Dockerfile`, `compose.yaml`; not yet built on a test machine): put the site's time zone (`TZ`, which zone `hours` follow), the camera addresses and the webhook in `.env`, create `logs/` and `status/`, then `docker compose up -d --build`. The image holds every model, so a site needs no internet, and runs on CPU or, with the NVIDIA Container Toolkit, a GPU. It leaves out the face-age model, which has no licence for commercial use. `uv run download-models --skip-age` does the same outside Docker.

Each camera process loads its own copy of the models: on a GPU server, check memory per camera before adding many.

## Face-based age (`--age`)

For eye-level cameras only. Age labels are Child, Adult, Older, and Unknown. The age model uses bands 0–2, 4–6, 8–12, 15–20, 25–32, 38–43, 48–53, and 60–100. The first three map to Child; 25–53 bands to Adult; 60–100 to Older. The 15–20 band stays Unknown because it crosses age 18. Each person's face is searched for within their own head region, about every third frame, and their label comes from the average of their last 15 face readings. Band scores are summed per group, and a group needs a total of at least `--age-conf` (default 0.6). Small (under 40 px), missing and ambiguous faces, and low scores, stay Unknown. Scores are not calibrated accuracy probabilities, and a stable label is not necessarily a correct one.

## Licences

Check these before commercial deployment:

| Component | Licence | Commercial use |
|---|---|---|
| RF-DETR Nano, Small, Medium and Large (person detector: code and weights) | Apache-2.0 | Yes. The Atto, Femto, Pico, XL and 2XL sizes use Roboflow's PML 1.0 licence and are not offered |
| trackers (ByteTrack), supervision | Apache-2.0, MIT | Yes |
| DINOv2-small (child classifier encoder) | Apache-2.0 | Yes |
| Open Images (child classifier training data) | Annotations CC BY 4.0; images listed as CC BY 2.0 | Yes, with attribution; verify each image's licence (saved in `data/child_crops/*.csv`) before redistributing crops |
| transformers | Apache-2.0 | Yes |
| Age model (Levi & Hassner, `--age`) | None stated | Not cleared |
| OpenCV res10 face detector (`--age`) | None stated for the weights | Unclear |

Video analytics on children is regulated in many places (for example GDPR, and biometric laws such as Illinois' BIPA); get a privacy and legal review before deployment.

## Model sources

Person detector: RF-DETR, https://github.com/roboflow/rf-detr. Tracker: ByteTrack from https://github.com/roboflow/trackers. Child classifier encoder: DINOv2, https://github.com/facebookresearch/dinov2 (https://huggingface.co/facebook/dinov2-small). Training data: Open Images, https://storage.googleapis.com/openimages/web/index.html. Age model: Gil Levi and Tal Hassner, *Age and Gender Classification Using Convolutional Neural Networks*, CVPR workshops 2015: https://github.com/GilLevi/AgeGenderDeepLearning. Only the age model is used. Face detector: https://github.com/opencv/opencv.

Weights, data and virtual environments are excluded from Git, except the small child classifier head. `download-models` fetches the RF-DETR, DINOv2 and age-model weights.
