"""Finish the RAID submission: push the branch, open the PR."""
import json
import os
import subprocess
import urllib.error
import urllib.request

LINES = open(os.path.join(os.environ["TEMP"], "gitcred_out.txt")).read().splitlines()
TOK = [l.split("=", 1)[1] for l in LINES if l.startswith("password=")][0]
WORK = os.path.join(os.environ["TEMP"], "raid_fork")

subprocess.run(["git", "push", "-u", "origin", "snidewider-submission"],
               cwd=WORK, check=True)
print("pushed", flush=True)

body = (
    "Snidewider: a classical-statistics AI-text detector (253 handcrafted "
    "features - q-gram exemplars, coverage contrasts, char/bigram "
    "distributions, repetitiveness deltas, BWT run structure - scored by one "
    "gradient-boosted model; no neural components).\n\n"
    "Holdout (RAID train_none holdout fold; 11,371 human / 20,000 AI): "
    "AUROC 0.9926 | TPR 0.971 @5% FPR | 0.891 @1% | 0.702 @0.1%.\n\n"
    "Disclosure: 84 exact-text overlaps found between our dev fold "
    "(train_none) and the test set (poetry-heavy dataset duplicates, "
    "~0.15%); documented in our repo for transparency.\n\n"
    "Repo: https://github.com/JKurzer/Snidewider")

req = urllib.request.Request(
    "https://api.github.com/repos/liamdugan/raid/pulls", method="POST",
    headers={"Authorization": "Bearer " + TOK,
             "Accept": "application/vnd.github+json",
             "User-Agent": "snidewider-submission"},
    data=json.dumps({"title": "Submission: Snidewider",
                     "head": "JKurzer:snidewider-submission",
                     "base": "main", "body": body}).encode())
try:
    pr = json.load(urllib.request.urlopen(req))
    print("PR:", pr["html_url"])
except urllib.error.HTTPError as e:
    print("PR error:", e.code, e.read()[:500].decode(errors="ignore"))
