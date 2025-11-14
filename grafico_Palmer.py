import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 

def plot_algorithm(filename, title, output_name):
    df = pd.read_csv(filename)
    df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
    grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
    grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
    colors = df['found'].map({True: 'lightblue', False: 'lightgray'})

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.title(title, fontsize=12, pad=10)

    ax.scatter(df['avg_degree'], df['time_s'], c=colors, alpha=0.6, s=10)
    ax.plot(grouped['degree_center'], grouped['time_s'], color='blue', linewidth=2)


    ax.set_xlabel("Average degree", fontsize=10)
    ax.set_ylabel("Execution time (s)", fontsize=10)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', length=0)

    yticks = ax.get_yticks()
    if len(yticks) > 0:
        ax.set_yticks(yticks[:-1])
        ax.set_yticklabels([f"{y:.0f}" for y in yticks[:-1]])

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.show()


# --- Exemplo de uso ---
plot_algorithm("results_palmer_20.csv", "Palmer", "grafico_palmer_unico.png")
