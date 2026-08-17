"""Analyze the 84 exact-text overlaps between RAID test_none and our dev fold."""
import pandas as pd

ours = pd.read_parquet("data/derived/raid_splits.parquet")
dev = ours[ours.fold == "dev"]
dev_hashes = {}
for _, row in dev.iterrows():
    dev_hashes.setdefault(hash(str(row.generation)), []).append((row.model, row.domain, row.source_id))

test = pd.read_csv(r"C:\Users\poly\.cache\raid\test_none.csv")
hits = []
for t in test.generation:
    h = hash(str(t))
    if h in dev_hashes:
        hits.extend(dev_hashes[h])

print(f"overlapping test rows: {len(hits)}")
if hits:
    o = pd.DataFrame(hits, columns=["model", "domain", "source_id"])
    print(o.model.value_counts().to_string())
    print("\ndomains:", o.domain.value_counts().to_dict())
    print("distinct dev sources involved:", o.source_id.nunique())
    # which bucket do these sources sit in?
    buckets = {}
    from ai_text_detection.evaldata import split_buckets
    bk = split_buckets(ours)
    for name in "ABC":
        for sid in set(o.source_id):
            if sid in set(bk[name].source_id):
                buckets.setdefault(name, 0)
                buckets[name] += 1
    print("bucket membership of involved sources:", buckets)
    # sample the overlapping texts
    texts = [str(t) for t in test.generation if hash(str(t)) in dev_hashes]
    for t in texts[:3]:
        print(f"\nSAMPLE (len {len(t)}): {t[:200]!r}")
