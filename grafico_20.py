# PARTE A
# 1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
fig.suptitle("Tempo de execução x grau médio", fontsize=14)
df = pd.read_csv("results_backtracking_20.csv")

# 10)
ax1 = fig.add_subplot(gs[0, 0])

# suponha que df tenha as colunas 'avg_degree' e 'time_s'
# 1. Criar faixas de avg_degree (30 bins)
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)

# 2. Agrupar por faixa e calcular média
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)

# 3. Calcular o centro de cada faixa (pra plotar no eixo X)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)

# 3.1. Definir cores com base em 'found'
colors = df['found'].map({True: 'lightblue', False: 'lightgray'})

# 4. Plotar tudo junto
ax1.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.6, s=10)

ax1.plot(grouped['degree_center'], grouped['time_s'],
         color='blue', linewidth=2, label='Média por faixa (30)')



ax1.set_xlabel("Average degree")
ax1.set_ylabel("Execution time (s)")
ax1.set_title("Backtracking")
plt.tight_layout()

df = pd.read_csv("results_palmer_20.csv")
ax2 = fig.add_subplot(gs[1, 1])
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
colors = df['found'].map({True: 'lightblue', False: 'lightgray'})
ax2.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.8, s=10)
ax2.plot(grouped['degree_center'], grouped['time_s'],
         color='blue', linewidth=2, label='Média por faixa (30)')

ax2.set_ylim(0, 22.5)
ax2.set_xlabel("Average degree")
ax2.set_ylabel("Execution time (s)")
ax2.set_title("Palmer")
plt.tight_layout()

df = pd.read_csv("results_vacul_20.csv")
ax3 = fig.add_subplot(gs[1, 0])
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
colors = df['found'].map({True: 'lightblue', False: 'lightgray'})
ax3.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.6, s=10)
ax3.plot(grouped['degree_center'], grouped['time_s'],
         color='blue', linewidth=2, label='Média por faixa (30)')


ax3.set_xlabel("Average degree")
ax3.set_ylabel("Execution time (s)")
ax3.set_title("VaCul")
plt.tight_layout()
fig.savefig("teste.png")

df = pd.read_csv("results_heldkarp_20.csv")
ax4 = fig.add_subplot(gs[0, 1])
df['degree_bin'] = pd.cut(df['avg_degree'], bins=30)
grouped = df.groupby('degree_bin', as_index=False).mean(numeric_only=True)
grouped['degree_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
colors = df['found'].map({True: 'lightblue', False: 'lightgray'})
ax4.scatter(df['avg_degree'], df['time_s'],
            c=colors, alpha=0.6, s=10)
ax4.plot(grouped['degree_center'], grouped['time_s'],
         color='blue', linewidth=2, label='Média por faixa (30)')

ax4.set_xlabel("Average degree")
ax4.set_ylabel("Execution time (s)")
ax4.set_title("Held-Karp")

found_true = plt.Line2D([], [], color='lightblue', marker='o', linestyle='None', label='Ciclo encontrado')
found_false = plt.Line2D([], [], color='lightgray', marker='o', linestyle='None', label='Não encontrado/timeout')
mean_line = plt.Line2D([], [], color='blue', linewidth=2, label='Média por faixa (30)')

fig.legend(
    handles=[found_true, found_false, mean_line],
    title='Legenda',
    loc='lower center',      # posição (pode ser 'lower center', 'right', etc.)
    ncol=2,                  # número de colunas na legenda
    frameon=True,
    fontsize=9,
    title_fontsize=10
)
plt.tight_layout()
fig.savefig("graficos_20.png")


########################