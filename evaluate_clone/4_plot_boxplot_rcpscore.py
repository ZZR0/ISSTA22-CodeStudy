import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon

# Your selected palettes for each model
COLOR_PALETTES = {
    "codebert": "cividis",
    "codet5": "inferno",
    "codegpt": "flare",
    "graphcodebert": "mako"
}

def generate_boxplot(jsonl_path: str, output_path: str):
    # Detect model to apply your specific palette
    model_path = Path(jsonl_path)
    model_name = model_path.parent.name.lower()
    palette_name = COLOR_PALETTES.get(model_name, "viridis")

    # 1. Read the JSONL data
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    df = pd.DataFrame(data)

    # 2. Perform Wilcoxon Signed-Rank Test
    # We need to pair the 'before' and 'after' scores for each project pair
    df_pivot = df.pivot(index='subject', columns='adapt', values='score').dropna()
    
    # Calculate Wilcoxon
    # H0: There is no median difference between Before and After
    stat, p_value = wilcoxon(df_pivot['before'], df_pivot['after'])

    # 3. Setup Plot
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="whitegrid")

    cmap = plt.get_cmap(palette_name)
    colors = [cmap(0.35), cmap(0.75)] 

    # 4. Create Boxplot
    ax = sns.boxplot(
        x='adapt', 
        y='score', 
        data=df, 
        order=['before', 'after'],
        palette=colors,
        width=0.5,
        linewidth=2,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
    )

    sns.stripplot(
        x='adapt', 
        y='score', 
        data=df, 
        order=['before', 'after'],
        color=".25", 
        alpha=0.4, 
        jitter=True
    )

    # 5. Titles and Labels
    formatted_model_name = model_name.replace("code", "Code").replace("gpt", "GPT").replace("bert", "BERT").replace("t5", "T5")
    plt.title(f'RCP Distribution: {formatted_model_name}', fontsize=16, pad=20)
    plt.xlabel('Adaptation Stage', fontsize=12, labelpad=10)
    plt.ylabel('RCP Score', fontsize=12, labelpad=10)
    
    # Add p-value annotation to the plot for the paper
    plt.annotate(f'Wilcoxon p-value: {p_value:.2e}', 
                 xy=(0.5, 0.05), xycoords='axes fraction', 
                 ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()

    # 6. Save and Print Statistics
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n--- Statistical Test Results for {formatted_model_name} ---")
    print(f"Wilcoxon Statistic: {stat:.3f}")
    print(f"P-Value: {p_value:.2e}")
    if p_value < 0.05:
        print("Result: Significant improvement (p < 0.05) ✅")
    else:
        print("Result: Not statistically significant (p >= 0.05) ❌")
    print(f"--> Saved Box Plot to: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Box Plot and run Wilcoxon test.")
    parser.add_argument("--rcp", type=str, required=True, help="Path to the rcp_scores.jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output box plot")
    
    args = parser.parse_args()
    
    if Path(args.rcp).exists():
        generate_boxplot(args.rcp, args.output)
    else:
        print(f"Error: Could not find '{args.rcp}'")