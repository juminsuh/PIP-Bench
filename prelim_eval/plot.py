import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

metrics = ["clip", "dino", "arcface", "fgis", "mcq_type1_baseline"]

fig, axes = plt.subplots(1, 5, figsize=(30, 5), sharey=False)
sns.set_style("whitegrid")

for i, metric in enumerate(metrics):
    file_path = f'/data1/joo/pai_bench/result/prelim_01/metric/content/{metric}.csv'
    
    df = pd.read_csv(file_path, dtype={'image0': str, 'image1': str})
    
    if metric == "mcq_type1_baseline":
        metric = "mcq_type1"
    score_col = f'{metric}_score'
    
    # ERROR 제외
    df = df[df[score_col] != "ERROR"].copy()
    df[score_col] = df[score_col].astype(float)
    
    # 중복 제거
    df['min_img'] = df[['image0', 'image1']].min(axis=1)
    df['max_img'] = df[['image0', 'image1']].max(axis=1)
    df = df.drop_duplicates(subset=['min_img', 'max_img'], keep='first').copy()
    
    # ID 부여
    def get_id(img_name):
        try:
            return (int(img_name) - 1) // 15
        except:
            return -1

    df['id0'] = df['image0'].apply(get_id)
    df['id1'] = df['image1'].apply(get_id)
    
    def assign_type(row):
        if row['image0'] == row['image1']:
            return 'Diagonal'
        elif row['id0'] == row['id1']:
            return 'Intra'
        else:
            return 'Inter'

    df['pair_type'] = df.apply(assign_type, axis=1)
    
    plot_df = df[df['pair_type'] != 'Diagonal'].copy()
    
    ax = axes[i]
    sns.histplot(data=plot_df, x=score_col, hue='pair_type',
                 bins=50, kde=True, stat="density", common_norm=False,
                 palette={'Intra': 'forestgreen', 'Inter': 'orange'},
                 alpha=0.4, ax=ax, legend=False)
    
    # 평균 수직선
    intra_mean = plot_df[plot_df['pair_type'] == 'Intra'][score_col].mean()
    inter_mean = plot_df[plot_df['pair_type'] == 'Inter'][score_col].mean()
    ax.axvline(intra_mean, color='darkgreen', linestyle='--', linewidth=1.5)
    ax.axvline(inter_mean, color='darkorange', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(-0.05, 1.05)

# 첫 번째 축에만 ylabel
axes[0].set_ylabel('Density', fontsize=12)

# 색 구분 안내 (첫 번째 subplot 상단)
axes[0].annotate('■ Intra', xy=(0.02, 0.95), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10, color='forestgreen')
axes[0].annotate('■ Inter', xy=(0.15, 0.95), xycoords='axes fraction',
                 ha='left', va='top', fontsize=10, color='orange')

plt.tight_layout()

output_dir = '/home/joo/minsuh/pai_bench/figs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'all_metrics_distribution.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 저장 완료: {save_path}")