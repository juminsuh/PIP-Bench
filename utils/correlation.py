import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

df = pd.read_csv("/data1/joo/pai_bench/result/prelim_02/human_clip.csv")

corr = df["human_score"].corr(df["clip_score"], method="pearson")
print(f"Pearson correlation: {corr}")

human = df["human_score"]
clip = df["clip_score"]

# Spearman correlation
rho, p_value = spearmanr(human, clip)

print(f"Spearman correlation (rho): {rho}")
print(f"P-value: {p_value}")


plt.figure(figsize=(6, 5))
plt.scatter(human, clip)

plt.xlabel("Human Score")
plt.ylabel("clip Score")
plt.title("Human Score vs clip Score")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fname="/home/joo/minsuh/pai_bench/figs/human_clip_figure.png", dpi=300, bbox_inches="tight")

