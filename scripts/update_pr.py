"""Update PR #184 with the full-coverage predictions.json."""
import os
import shutil
import subprocess

WORK = os.path.join(os.environ["TEMP"], "raid_fork")

shutil.copy("submissions/snidewider/predictions.json",
            os.path.join(WORK, "leaderboard", "submissions", "snidewider",
                         "predictions.json"))
size = os.path.getsize(os.path.join(
    WORK, "leaderboard", "submissions", "snidewider", "predictions.json"))
print(f"predictions.json updated ({size/1e6:.1f} MB)", flush=True)

subprocess.run(["git", "add", "leaderboard/submissions/snidewider/predictions.json"],
               cwd=WORK, check=True)
subprocess.run(["git", "-c", "user.name=JKurzer",
                "-c", "user.email=jkurzer@users.noreply.github.com",
                "commit", "-m",
                "Full-coverage predictions (672,000 rows: all domains, "
                "generators, decoding strategies, repetition penalties, and "
                "adversarial attacks)"], cwd=WORK, check=True)
subprocess.run(["git", "push"], cwd=WORK, check=True)
print("pushed - PR #184 updated", flush=True)
