"""Patch PR #184's body to match the 261-champion payload."""
import json
import os
import subprocess
import urllib.request

cred_in = os.path.join(os.environ["TEMP"], "gitcred_in.txt")
with open(cred_in, "w") as fh:
    fh.write("protocol=https\nhost=github.com\n\n")
out = subprocess.run(["git", "credential", "fill"], stdin=open(cred_in),
                     capture_output=True, text=True).stdout
os.remove(cred_in)
tok = [l.split("=", 1)[1] for l in out.splitlines() if l.startswith("password=")][0]

body = (
    "Snidewider: a classical-statistics AI-text detector (261 handcrafted "
    "features - q-gram exemplars, coverage contrasts, char/bigram "
    "distributions, repetitiveness deltas, BWT run structure, self-anchor "
    "colinear-chain statistics - scored by one gradient-boosted model "
    "(depth 8, 600 iters); no neural components).\n\n"
    "Holdout (RAID train_none holdout fold; 11,371 human / 20,000 AI): "
    "AUROC 0.9935 | TPR 0.974 @5% FPR | 0.912 @1% | 0.754 @0.1%.\n\n"
    "Disclosure: 84 exact-text overlaps found between our dev fold "
    "(train_none) and the test set (poetry-heavy dataset duplicates, "
    "~0.15%); documented in our repo for transparency.\n\n"
    "Full-coverage predictions (672,000 rows: all domains, generators, "
    "decoding strategies, repetition penalties, adversarial attacks).\n\n"
    "Repo: https://github.com/JKurzer/Snidewider")

req = urllib.request.Request(
    "https://api.github.com/repos/liamdugan/raid/pulls/184", method="PATCH",
    headers={"Authorization": "Bearer " + tok,
             "Accept": "application/vnd.github+json",
             "User-Agent": "snidewider-submission"},
    data=json.dumps({"body": body}).encode())
pr = json.load(urllib.request.urlopen(req))
print("PR body patched:", pr["html_url"])
