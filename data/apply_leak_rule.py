# # Re-apply a stricter leak rule to an existing leak-check run (no re-embedding)
#
# `run_leak_check` (2026-11-05) dropped 14,001 home-decore train rows with the first rule. Looking at the pairs:
# identical DF scene renders scored 0.974 and 0.999, while different products (two doors, two wardrobes,
# two living rooms) scored 0.950-0.964. So 0.95 is too loose for catalog photos and renders.
#
# `leak_rows.jsonl` already holds every score for every image, so this script only re-reads it, applies
# the new rule and saves `exclude_v2.txt` next to the old files. No GPU, no HF token, a few minutes.
#
# New rule, an image counts as a copy of an eval image if:
# - identical pixels, or
# - near-copy fingerprint (dhash <= 6 bits) and SigLIP similarity of that pair >= `DHASH_PAIR_SIM`, or
# - SigLIP similarity >= `EMBED_SIM` (stricter: no fingerprint agreement)


import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "finetuning")]
from fsutil import fs, show_html   # file helpers and HTML review pages
# %% ---------------------------------------------------------------------------

# 1. Settings
import collections, json

CHECK_DIR = "decore/checks/leak/20261105T055610Z"
SOURCES = ["home-decore", "pix3d", "coco_scenes"]
DHASH_BITS = 6
DHASH_PAIR_SIM = 0.97      # identical DF scenes: 0.974 / 0.999; two different doors: 0.964
EMBED_SIM = 0.98           # different products reached 0.95-0.971 (two different lamps: 0.971)
# DF scenes: design variants of the same room (extra light, other door colour) sit at 0.975-0.98 and are leaks
# for scene retrieval; different rooms were ~0.95. So scenes use a lower bar.
EMBED_SIM_BY_CONFIG = {"deepfurniture_scenes": 0.97}

# %% ---------------------------------------------------------------------------

# 2. Apply the rule per source and save exclude_v2.txt + leak_summary_v2.json


def read_text(path):
    return fs.head(path, fs.ls(path)[0].size)


def is_copy(r):
    if r["dropped"] == "same_pixels":
        return "same_pixels"
    if r["dhash_bits"] <= DHASH_BITS and r.get("dhash_pair_sim", 0) >= DHASH_PAIR_SIM:
        return "dhash"
    config = r["name"].split("/")[0]
    if r["max_sim"] >= EMBED_SIM_BY_CONFIG.get(config, EMBED_SIM):
        return "embedding"
    return ""


overall = {}
for source in SOURCES:
    rows = [json.loads(l) for l in read_text(f"{CHECK_DIR}/{source}/leak_rows.jsonl").splitlines() if l]
    copies = [(r, reason) for r in rows if (reason := is_copy(r))]
    exclude = sorted({r["name"].split("#")[0] for r, _ in copies})
    by_reason = collections.Counter(reason for _, reason in copies)
    by_eval = collections.Counter(r["nearest_eval"].split(":")[0].split("/")[0] for r, _ in copies)
    by_train = collections.Counter(r["name"].split("/")[0] if "/" in r["name"] else source for r, _ in copies)
    summary = {"source": source, "checked": len(rows), "old_rule_dropped": sum(1 for r in rows if r["dropped"]),
               "new_rule_copies": len(copies), "rows_excluded": len(exclude),
               "by_reason": dict(by_reason), "by_eval_set": dict(by_eval), "by_train_config": dict(by_train),
               "rule": {"dhash_bits": DHASH_BITS, "dhash_pair_sim": DHASH_PAIR_SIM, "embed_sim": EMBED_SIM,
                        "embed_sim_by_config": EMBED_SIM_BY_CONFIG}}
    overall[source] = summary
    assert fs.put(f"{CHECK_DIR}/{source}/exclude_v2.txt", "\n".join(exclude) + "\n", overwrite=True)
    assert fs.put(f"{CHECK_DIR}/{source}/leak_summary_v2.json", json.dumps(summary, indent=2), overwrite=True)

assert fs.put(f"{CHECK_DIR}/summary_v2.json", json.dumps(overall, indent=2), overwrite=True)
print(json.dumps(overall, indent=2))
