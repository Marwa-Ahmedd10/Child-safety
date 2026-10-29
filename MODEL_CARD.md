# Model card: child/adult classifier and person detector

Version: `child_head c2c695839fd9` (first 12 hex digits of `models/child_head.npz`'s SHA-256, trained 2026-10-27) with detector `rf-detr-nano`. Every alert records the versions it was made with, in its `models` field. Update this card whenever either changes.

## What it does

The **person detector** (RF-DETR Nano, Apache-2.0, COCO-pretrained) finds people in each frame; ByteTrack follows them. The **child classifier** turns each tracked person's picture into a probability of being a child: a frozen DINOv2-small image encoder (Apache-2.0) and a logistic-regression head trained here. A person's label is the average of their last 10 readings: Child above a threshold, Adult below, Checking until 3 readings. People under 64 px tall are not judged. `--child-recall` picks the threshold from the share of children it should find.

## Intended use

To prompt staff to look: alerts for possible children in hazard zones (any area marked as one, such as stairs, a kitchen, a car park or water), children alone at exits, and child limits per area. **Not** for decisions about an individual, age verification, identification or counting for billing, and not as the only safeguard. It infers apparent age group only; there is no face recognition, identity or gender inference, and no images are stored.

## Training data

1,254 person crops from Open Images (annotations CC BY 4.0, images listed as CC BY 2.0): 627 children ("Boy"/"Girl" boxes from photos people verified as showing a child, toddler or baby) and 627 adults ("Man"/"Woman" boxes from photos not verified as showing one), plus 3 copies of each degraded to look like webcam or CCTV images. Mostly eye-level photos. About 5–10% of labels look wrong, mostly teenagers labelled as adults.

## Measured performance

Held-out Open Images photos (514 people, half children), single readings, children found / adults flagged as children:

| `--child-recall` | Photos | People 64 px tall | Webcam-quality copies |
|---|---|---|---|
| 0.95 (default) | 96% / 19% | 96% / 30% | 92% / 32% |
| 0.90 | 93% / 11% | 93% / 18% | 88% / 18% |
| 0.85 | 88% / 7% | 89% / 11% | 80% / 11% |
| 0.80 | 83% / 4% | 84% / 5% | 71% / 7% |

Cross-validated AUC 0.951.

**From above**, on a mall clip filmed from an upper floor (98 people, all or nearly all adults, per-person averages): 85% of adults flagged as children at 0.95, 49% at 0.90, 16% at 0.85, 2% at 0.80. How many children each setting finds from above has **not been measured**: it needs labelled footage with children from the real cameras.

**Person detector:** 48.4 COCO AP (published; YOLO11n, which it replaced, scores 39.5). On the mall clip it found 14 people in a frame where YOLO11n found 6, all real, but neither found the small, distant people packed on escalators. 70 ms per frame on an M1 MacBook Air (8 GB), 2.3 ms on an NVIDIA T4 (published).

## Known failure modes

- **High and ceiling cameras:** adults' heads look bigger and legs shorter from above, so many adults are labelled Child at the default setting.
- **Partly hidden people:** with only their upper body showing, adults score as children. Measured in water; expect the same behind counters, railings or parked cars, which has not been measured. In a generated pool video, two adults standing in the water scored 0.45–0.47 (Child at `--child-recall` 0.95 and 0.90, one of them still at 0.85), while the child walking past them dipped as low as 0.32, so no setting separated them. Swimmers showing only their heads are too small to judge at all.
- **Older children and swimwear:** a boy of about 10–12 in swimwear at a pool edge scored 0.10–0.24, Adult at every setting. Where nobody should be at certain times, such as a pool, kitchen or plant room after hours, use an `"anyone"` hazard zone with `"hours"`, which does not depend on this classifier.
- **Teenagers** are inherently ambiguous; "child" means visibly a child.
- **Not evaluated across** skin tones, body types, disabilities or clothing such as abayas and hijabs. Measure on site footage before relying on it there.
- **Tracking:** in crowds an ID can pass from one person to another, and someone leaving the view for more than about 2 seconds returns as a new person, starting at Checking.

## Updating

Record footage from cameras mounted at the height and angle the product will use (see the team's recording guide; any room will do, as the classifier judges people, not places), label it with `uv run footage label`, measure the current model with `uv run footage evaluate`, retrain with `uv run train-child-classifier --site data/site_crops --test-session <held-back session>`, and update this card's version and numbers from the held-back session before deploying the new `child_head.npz`.
