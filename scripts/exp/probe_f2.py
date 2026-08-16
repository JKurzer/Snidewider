"""Probe: schema + token-length distribution by class on the dev fold."""

import pandas as pd

df = pd.read_parquet(r"C:\Users\poly\ai-text-detection\data\derived\raid_splits.parquet")
print("columns:", list(df.columns))
print("folds:", df.fold.value_counts().to_dict())

dev = df[df.fold == "dev"]
humans = dev[dev.model == "human"]
ai = dev[dev.model != "human"].sample(n=4000, random_state=17)
print(f"dev humans={len(humans)} ai_sample={len(ai)}")

tok_h = humans.generation.astype(str).str.split().str.len()
tok_a = ai.generation.astype(str).str.split().str.len()

bins = [0, 100, 200, 300, 400, 500, 600, 800, 10**9]
labels = ["<100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-800", "800+"]
tab = pd.DataFrame(
    {
        "human": pd.cut(tok_h, bins, labels=labels, right=False).value_counts().sort_index(),
        "ai": pd.cut(tok_a, bins, labels=labels, right=False).value_counts().sort_index(),
    }
)
print(tab.to_string())
print(f"\nhumans >=500 tok: {(tok_h >= 500).sum()} | ai >=500 tok: {(tok_a >= 500).sum()}")
print(f"human median={tok_h.median():.0f} mean={tok_h.mean():.0f} | ai median={tok_a.median():.0f} mean={tok_a.mean():.0f}")

# source_id uniqueness check (split unit)
print("\nsource overlap human/ai:", len(set(humans.source_id) & set(ai.source_id)))
print("unique sources humans:", humans.source_id.nunique(), "| ai:", ai.source_id.nunique())
