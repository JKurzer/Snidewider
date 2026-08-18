"""Hand-verify oct_repeat_features against worked examples."""
from ai_text_detection.token_bigrams import oct_repeat_features

# 1. no repeats at all -> 0
t1 = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
r1 = oct_repeat_features(t1)
print(f"no-repeat text:        rate {r1['oct_repeat_rate']:.3f} (expect 0.000)")

# 2. worked example: 'the cat sat on the mat sat on the porch'
#    the x3, sat x2, on x2 all within +/-4 of each other -> 6 hits / 10 tokens
t2 = "the cat sat on the mat sat on the porch"
r2 = oct_repeat_features(t2)
print(f"worked example:        rate {r2['oct_repeat_rate']:.3f} (expect 0.600), "
      f"abs {r2['oct_repeat_abs']:.0f} (expect 6)")

# 3. long-range repeat (beyond +/-4) should NOT count
t3 = "dog " + "one two three four five six seven eight " + "dog"
r3 = oct_repeat_features(t3)
print(f"long-range repeat:     rate {r3['oct_repeat_rate']:.3f} (expect 0.000 - "
      f"the two 'dog's are 9 apart)")

# 4. adjacent repeat (immediately neighboring) counts
t4 = "the the quick brown fox jumps over a lazy dog somewhere nearby today"
r4 = oct_repeat_features(t4)
print(f"adjacent repeat:       rate {r4['oct_repeat_rate']:.3f} (expect ~0.143 = 2/14)")

# 5. real text sanity: AI-ish boilerplate vs plain
t5 = ("It is important to note that it is essential to consider the fact that "
      "it is crucial to understand the importance of the matter at hand today")
r5 = oct_repeat_features(t5)
print(f"boilerplate:           rate {r5['oct_repeat_rate']:.3f} "
      f"(expect high - 'it/is/to/the' recur locally)")
