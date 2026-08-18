"""Print the top-8 (keepers) and bottom-20 (noise candidates) of the knockout table."""
lines = open("docs/exp/fleet_knockout.md", encoding="utf-8").readlines()
rows = [l for l in lines if l.startswith("| ") and not l.startswith("| feature") and not l.startswith("|---")]
print("=== top 8 (most valuable; removing them costs most) ===")
print("".join(rows[:8]))
print("=== bottom 20 (noise/dilution candidates) ===")
print("".join(rows[-20:]))
