# Results

All tests use data that no model trained on and that was not used to pick checkpoints. Differences come with
95% bootstrap intervals; an interval that excludes 0 is a real difference. **Final model: `v3-clean-nostyle`.**

## Models compared

| Model | Training data | Notes |
|---|---|---|
| `base` | none | `google/siglip2-base-patch16-224` |
| `v3-baseline` | ABO + DeepFurniture after the leak check only | checkpoint picked on Shop the Look selection queries (step 750) |
| `v3-clean` | cleaned data + new captions / queries, DeepFurniture style label kept | picked on Shop the Look (step 500); `@1476` = last step |
| **`v3-clean-nostyle`** | cleaned data + new captions / queries, no one-word style label | **final**; picked on a validation score (step 1476), no Shop the Look |
| `v3-clean-oi` | as nostyle + 13.5k cleaned Open Images real-photo crops and 9k photos (15% of each batch) | experiment |

All use the same recipe: full fine-tune, 1,476 steps, batch 128, learning rate 5e-6, 1.25 to 1.5 h on one A10G.
An earlier run with Pix3D + COCO real photos added (`v3-real_photos`) matched `v3-baseline` within noise on the
Shop the Look selection queries and is not tested here.

## 1. Real room photo → product (Shop the Look)

2,250 held-out queries (crops of real room photos), gallery of 19,253 products. "Correct" products are often
similar items rather than the exact one, so absolute scores understate every model.

| Model | R@1 | R@10 | R@50 | MRR |
|---|---|---|---|---|
| base | 0.076 | 0.364 | 0.682 | 0.174 |
| v3-baseline | 0.147 | 0.500 | 0.749 | 0.267 |
| v3-clean (step 500) | 0.141 | 0.537 | 0.770 | 0.266 |
| v3-clean (step 1476) | 0.161 | 0.507 | 0.772 | 0.272 |
| **v3-clean-nostyle** | **0.164** | 0.508 | **0.774** | **0.274** |
| v3-clean-oi | 0.149 | 0.504 | 0.762 | 0.262 |

| Difference | R@1 | R@10 |
|---|---|---|
| nostyle − base | +0.088 [+0.073, +0.103] | +0.144 [+0.124, +0.164] |
| v3-baseline − nostyle | −0.017 [−0.033, −0.003] | −0.008 [−0.021, +0.007] |
| oi − nostyle | −0.015 [−0.028, −0.003] | −0.004 [−0.016, +0.008] |

Per category, R@1 / R@10 (about 200 queries each, so differences under about 0.06 are within noise):

| Category | base | v3-baseline | nostyle | oi |
|---|---|---|---|---|
| ceiling light | 0.130 / 0.481 | 0.226 / 0.553 | 0.178 / 0.572 | 0.245 / 0.620 |
| chair | 0.094 / 0.401 | 0.175 / 0.561 | 0.160 / 0.519 | 0.165 / 0.561 |
| curtain | 0.028 / 0.164 | 0.023 / 0.360 | 0.028 / 0.271 | 0.047 / 0.280 |
| decor | 0.086 / 0.299 | 0.117 / 0.442 | 0.122 / 0.472 | 0.102 / 0.467 |
| faucet | 0.033 / 0.292 | 0.156 / 0.462 | 0.142 / 0.547 | 0.118 / 0.425 |
| lamp | 0.198 / 0.615 | 0.219 / 0.714 | 0.323 / 0.703 | 0.266 / 0.807 |
| mirror | 0.045 / 0.261 | 0.121 / 0.412 | 0.121 / 0.412 | 0.141 / 0.402 |
| rug | 0.028 / 0.377 | 0.198 / 0.637 | 0.264 / 0.642 | 0.208 / 0.613 |
| sofa | 0.049 / 0.300 | 0.113 / 0.355 | 0.069 / 0.379 | 0.044 / 0.350 |
| stool | 0.146 / 0.624 | 0.205 / 0.712 | 0.298 / 0.766 | 0.229 / 0.707 |
| throw pillow | 0.010 / 0.194 | 0.061 / 0.291 | 0.107 / 0.306 | 0.077 / 0.316 |

## 2. Text search on held-out images (test split)

Captions and 3 search queries per test image were written by gpt-4.1-mini, the same way as the training text,
so the fine-tuned models have a style advantage here; compare models with each other. R@1 / R@10:

| Test | Queries | base | v3-baseline | nostyle | oi |
|---|---|---|---|---|---|
| DeepFurniture scene: caption → room | 1,290 | 0.539 / 0.847 | 0.602 / 0.896 | 0.839 / 0.987 | 0.832 / 0.988 |
| DeepFurniture scene: query → room | 3,870 | 0.063 / 0.233 | 0.056 / 0.245 | 0.090 / 0.334 | 0.087 / 0.330 |
| DeepFurniture crop: caption → crop | 2,559 | 0.211 / 0.686 | 0.249 / 0.746 | 0.297 / 0.814 | 0.306 / 0.815 |
| DeepFurniture crop: query → crop | 7,677 | 0.075 / 0.361 | 0.079 / 0.391 | 0.100 / 0.440 | 0.098 / 0.444 |
| ABO: query → product photo | 771 | 0.620 / 0.975 | 0.646 / 0.984 | 0.680 / 0.984 | 0.660 / 0.984 |

nostyle − base, R@10: scene caption +0.140 [+0.120, +0.157], scene query +0.101 [+0.088, +0.115], crop caption
+0.128 [+0.111, +0.145], crop query +0.080 [+0.069, +0.090], ABO query +0.009 [+0.000, +0.018].
oi − nostyle: every interval includes 0.

### Scene leak check

DeepFurniture renders each room from several cameras and its split is not by room. Looking at test rooms next to
their most similar training room (base SigLIP 2 similarity): 2 of 2 pairs at ≥ 0.97, 6 of 8 at 0.95 to 0.97 and
1 of 4 at 0.93 to 0.95 were the same room from another angle; none was seen below 0.93. Re-scoring only test
rooms whose nearest training room is below each cut:

| Test rooms kept | Queries | base R@10 | v3-baseline | nostyle | nostyle − base |
|---|---|---|---|---|---|
| all | 3,870 | 0.233 | 0.245 | 0.334 | +0.101 [+0.088, +0.115] |
| nearest < 0.95 | 2,667 | 0.243 | 0.255 | 0.341 | +0.098 [+0.081, +0.114] |
| **nearest < 0.93 (no repeated rooms seen)** | 1,428 | 0.281 | 0.312 | **0.391** | **+0.110 [+0.090, +0.131]** |
| nearest < 0.90 | 432 | 0.317 | 0.373 | 0.428 | +0.111 [+0.069, +0.153] |

The gain holds on rooms with no copy in training, so it is not explained by the leak. v3-baseline saw the same
training rooms and gained little.

## 3. Real furniture photo → same furniture model (Pix3D)

6,795 queries; correct = any other photo of the same 3D model. Never used for training by these models.

| Model | R@1 | R@10 | MRR |
|---|---|---|---|
| base | 0.778 | 0.959 | 0.843 |
| v3-baseline | 0.816 | 0.973 | 0.872 |
| **nostyle** | **0.827** | **0.976** | **0.881** |
| oi | 0.820 | 0.974 | 0.876 |

nostyle − base R@1 +0.049 [+0.041, +0.057]; oi − nostyle R@1 −0.006 [−0.012, −0.001].

| Category (R@1) | Queries | base | v3-baseline | nostyle | oi |
|---|---|---|---|---|---|
| bed | 528 | 0.725 | 0.801 | 0.794 | 0.792 |
| bookcase | 212 | 0.627 | 0.726 | 0.750 | 0.722 |
| chair | 3,010 | 0.902 | 0.922 | 0.930 | 0.930 |
| desk | 421 | 0.658 | 0.777 | 0.781 | 0.748 |
| sofa | 1,331 | 0.630 | 0.644 | 0.664 | 0.660 |
| table | 1,113 | 0.700 | 0.761 | 0.775 | 0.762 |
| wardrobe | 180 | 0.883 | 0.889 | 0.906 | 0.894 |

## 4. Everyday captions of real home photos (COCO)

Home photos only (caption word rule), 5 human captions each: 5,564 caption → photo queries.

| Model | R@1 | R@10 | MRR |
|---|---|---|---|
| **base** | **0.461** | **0.784** | **0.570** |
| v3-baseline | 0.388 | 0.714 | 0.497 |
| nostyle | 0.436 | 0.771 | 0.550 |
| oi | 0.395 | 0.732 | 0.507 |

nostyle − base R@1 −0.024 [−0.035, −0.014]; v3-baseline − base −0.072 [−0.083, −0.061]; oi − nostyle −0.042
[−0.051, −0.032]. Cleaning the data cut the loss against base from 7.2 to 2.4 points. Many COCO captions describe
people and activities, which this test does not separate.

## 5. Room type on real photos (Places365)

Zero-shot classification among 22 home room types, 100 validation photos each (2,200), short prompts such as
"a photo of a bedroom". Top-1 / top-3:

| Model | Top-1 | Top-3 |
|---|---|---|
| **base** | **0.691** | **0.894** |
| v3-baseline | 0.640 | 0.872 |
| nostyle | 0.649 | 0.878 |
| oi | 0.671 | 0.888 |

nostyle − base −0.042 [−0.056, −0.026]; oi − nostyle +0.022 [+0.011, +0.033].

| Room type (top-1) | base | nostyle | oi |
|---|---|---|---|
| bedroom | 0.79 | 0.77 | 0.83 |
| living room | 0.56 | 0.63 | 0.70 |
| kitchen | 0.86 | 0.89 | 0.92 |
| dining room | 0.76 | 0.79 | 0.81 |
| bathroom | 0.92 | 0.91 | 0.93 |
| home office | 0.82 | 0.69 | 0.78 |
| nursery | 0.91 | 0.92 | 0.87 |
| kids room | 0.29 | 0.36 | 0.35 |
| closet | 0.86 | 0.70 | 0.74 |
| dorm room | 0.57 | 0.46 | 0.51 |
| attic | 0.73 | 0.72 | 0.71 |
| basement | 0.54 | 0.29 | 0.40 |
| corridor | 0.80 | 0.83 | 0.85 |
| pantry | 0.79 | 0.80 | 0.83 |
| utility room | 0.73 | 0.72 | 0.71 |
| TV room | 0.56 | 0.38 | 0.42 |
| playroom | 0.60 | 0.49 | 0.55 |
| home theater | 0.51 | 0.49 | 0.44 |
| staircase | 0.93 | 0.90 | 0.92 |
| entrance hall | 0.52 | 0.46 | 0.47 |
| home bar | 0.31 | 0.24 | 0.22 |
| indoor garage | 0.83 | 0.83 | 0.80 |

Room types common in the DeepFurniture training renders (living room, kids room, dining room, kitchen) improved;
rare ones (basement, TV room, closet, home office) dropped. Open Images photos (patios, offices, hotel rooms)
recovered about half of that loss.

## What the experiments show

- Cleaning the data and writing new captions and queries gave the large text-search gains and the best Shop the
  Look R@1; it also reduced the loss on everyday captions (COCO) from 7.2 to 2.4 points.
- Real photos paired only with text (Pix3D + COCO, then 13.5k cleaned Open Images photos) did not improve real
  photo → product search. They improved room types but cost Shop the Look R@1 and COCO. Shop the Look matches a
  real photo to a catalog photo; text-only real photos do not teach that cross-domain image match.
- Remaining gap: rare room types and everyday wording, both below base. Weight averaging with the base model
  (WiSE-FT, Wortsman et al., 2022) is the untested next step.
