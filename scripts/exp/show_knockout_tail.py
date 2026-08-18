"""Re-rank the knockout:1 table by dTPR@1e-3 (the long-tail column)."""
lines = open("docs/exp/fleet_knockout.md", encoding="utf-8").readlines()
rows = []
for l in lines:
    if l.startswith("| ") and not l.startswith(("| feature", "|---")):
        parts = [p.strip() for p in l.strip().strip("|").split("|")]
        rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))

rows.sort(key=lambda r: r[3], reverse=True)
print("=== features whose removal IMPROVES the 1e-3 tail (tail-hurters) ===")
print("| feature | dAUROC | dTPR@1e-2 | dTPR@1e-3 |")
for name, dr, dt1, dt3 in rows[:22]:
    print(f"| {name} | {dr:+.4f} | {dt1:+.3f} | {dt3:+.3f} |")
print("\n=== features whose removal COSTS the tail most (tail-earners) ===")
for name, dr, dt1, dt3 in rows[-8:]:
    print(f"| {name} | {dr:+.4f} | {dt1:+.3f} | {dt3:+.3f} |")
