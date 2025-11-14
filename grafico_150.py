import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
fig.suptitle("Tempo de execução x grau médio", fontsize=14)
df = pd.read_csv("results_backtracking.csv")

ax1 = fig.add_subplot(gs[0, 0])
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
colors = df['found'].map({True: 'blue', False: 'lightgray'})
ax1.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.6, s=10)


ax1.set_xlabel("Average degree")
ax1.set_ylabel("Execution time (s)")
ax1.set_title("Backtracking")
plt.tight_layout()

df = pd.read_csv("results_palmer.csv")
ax2 = fig.add_subplot(gs[1, 1])
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
colors = df['found'].map({True: 'blue', False: 'lightgray'})
ax2.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.8, s=10)

ax2.set_ylim(0, 22.5)
ax2.set_xlabel("Average degree")
ax2.set_ylabel("Execution time (s)")
ax2.set_title("Palmer")
plt.tight_layout()

df = pd.read_csv("results_vacul.csv")
ax3 = fig.add_subplot(gs[1, 0])
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
colors = df['found'].map({True: 'blue', False: 'lightgray'})
ax3.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.6, s=10)


ax3.set_xlabel("Average degree")
ax3.set_ylabel("Execution time (s)")
ax3.set_title("VaCul")
plt.tight_layout()
fig.savefig("teste.png")

df = pd.read_csv("results_heldkarp.csv")
ax4 = fig.add_subplot(gs[0, 1])
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
colors = df['found'].map({True: 'blue', False: 'lightgray'})
ax4.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.6, s=10)

ax4.set_xlabel("Average degree")
ax4.set_ylabel("Execution time (s)")
ax4.set_title("Held-Karp")

found_true = plt.Line2D([], [], color='blue', marker='o', linestyle='None', label='Ciclo encontrado')
found_false = plt.Line2D([], [], color='lightgray', marker='o', linestyle='None', label='Não encontrado/timeout')

fig.legend(
    handles=[found_true, found_false],
    title='Legenda',
    loc='lower center',      
    ncol=2,                 
    frameon=True,
    fontsize=9,
    title_fontsize=10
)
plt.tight_layout()
fig.savefig("graficos_150.png")
