"""Fire the RAID submission: fork, verify layout, clone, add files, push, PR.

Token from the Git Credential Manager (never printed). Steps are idempotent
and verbose; each stage reports before proceeding.
"""

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

LINES = open(r"%TEMP%\gitcred_out.txt".replace("%TEMP%", __import__("os").environ["TEMP"])).read().splitlines()
TOK = [l.split("=", 1)[1] for l in LINES if l.startswith("password=")][0]
OWNER = "JKurzer"
UPSTREAM = "liamdugan/raid"
WORK = Path(__import__("os").environ["TEMP"]) / "raid_fork"


def gh(method: str, path: str, data: dict | None = None) -> dict:
    req = urllib.request.Request(
        "https://api.github.com" + path, method=method,
        headers={"Authorization": "Bearer " + TOK,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "snidewider-submission"},
        data=json.dumps(data).encode() if data else None)
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read()[:400].decode(errors="ignore")}


def main() -> None:
    print("== 1. fork ==", flush=True)
    r = gh("POST", f"/repos/{UPSTREAM}/forks")
    print("  fork:", r.get("full_name", r), flush=True)
    for _ in range(30):  # fork creation is async
        chk = gh("GET", f"/repos/{OWNER}/raid")
        if "error" not in chk:
            break
        time.sleep(4)
    print("  fork ready:", chk.get("full_name", chk), flush=True)

    print("== 2. layout check (an existing submission) ==", flush=True)
    entries = gh("GET", f"/repos/{UPSTREAM}/contents/leaderboard/submissions")
    names = [e["name"] for e in entries if e["type"] == "dir"]
    print("  submissions dirs:", names[:6], flush=True)
    sample = gh("GET", f"/repos/{UPSTREAM}/contents/leaderboard/submissions/{names[0]}")
    print(f"  files in {names[0]}:", [e["name"] for e in sample], flush=True)

    print("== 3. clone + place files ==", flush=True)
    if not WORK.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://x-access-token:{TOK}@github.com/{OWNER}/raid.git",
                        str(WORK)], check=True, capture_output=True)
    dest = WORK / "leaderboard" / "submissions" / "snidewider"
    dest.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in ("predictions.json", "metadata.json"):
        shutil.copy(f"submissions/snidewider/{f}", dest / f)
        print(f"  placed {f} ({(dest / f).stat().st_size} bytes)", flush=True)

    print("== 4. branch + push ==", flush=True)
    subprocess.run(["git", "checkout", "-b", "snidewider-submission"],
                   cwd=WORK, check=True, capture_output=True)
    subprocess.run(["git", "add", "leaderboard/submissions/snidewider"],
                   cwd=WORK, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=JKurzer", "-c",
                    "user_email=jkurzer@users.noreply.github.com",
                    "commit", "-m",
                    "Add Snidewider submission (classical-statistics detector, "
                    "250+ features, no neural components)"],
                   cwd=WORK, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "snidewider-submission"],
                   cwd=WORK, check=True, capture_output=True)
    print("  pushed", flush=True)

    print("== 5. open PR ==", flush=True)
    pr = gh("POST", f"/repos/{UPSTREAM}/pulls", {
        "title": "Submission: Snidewider",
        "head": f"{OWNER}:snidewider-submission",
        "base": "main",
        "body": ("Snidewider: a classical-statistics AI-text detector "
                 "(250+ handcrafted features — q-gram exemplars, coverage, "
                 "char/bigram distributions, repetitiveness deltas, BWT run "
                 "structure — scored by a single gradient-boosted model; no "
                 "neural components).\n\n"
                 "Disclosure: 84 exact-text overlaps found between our dev "
                 "fold (RAID train_none) and the test set (poetry-heavy "
                 "dataset duplicates, ~0.15%); documented in our repo.\n\n"
                 "Holdout (train_none holdout fold, 11,371 hu / 20,000 ai): "
                 "AUROC 0.9926, TPR 0.971 @5%FPR / 0.891 @1% / 0.702 @0.1%."),
    })
    print("  PR:", pr.get("html_url", pr), flush=True)


if __name__ == "__main__":
    main()
