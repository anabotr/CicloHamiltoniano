### Objetivo
O objetivo deste repositório é complementar o trabalho para a A2 sobre o problema do ciclo hamiltoniano em grafos, avaliando desempenho, complexidade e comportamento prático dos algoritmos selecionados na resolução da questão porposta por meio de experimentos e visualizações gráficas.


### Arquivos principais
| Arquivo             | Descrição                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| **backtracking.py** | Implementação do algoritmo de backtracking.                               |
| **heldkarp.py**     | Implementação do algoritmo de Held-Karp.                  |
| **palmer.py**       | Implementação do algoritmo de Palmer.                                         |
| **vacul.py**        | Implementação do algoritmo VaCul.                                     |

### Scripts de geração de gráficos
| Arquivo               | Descrição                                                                        |
| --------------------- | -------------------------------------------------------------------------------- |
| **grafico_20.py**     | Gera a imagem com 4 gráficos a partir dos resultados (.csv) dos testes com 20 vértices. |
| **grafico_150.py**    | Gera a imagem com 4 gráficos a partir dos resultados (.csv) dos testes para até 150 vértices.                        |
| **grafico_Palmer.py** | Gera o gráfico único dos resultados de Palmer para 20 vértices.               |

### Arquivos de resultados (.csv)
| Arquivo                         | Descrição                                                              |
| ------------------------------- | ---------------------------------------------------------------------- |
| **results_backtracking.csv**    | Resultados completos do algoritmo backtracking para até 150 vértices. |
| **results_backtracking_20.csv** | Resultados do backtracking para 20 vértices.                 |
| **results_heldkarp.csv**        | Resultados do Held-Karp para até 150 vértices.                      |
| **results_heldkarp_20.csv**     | Resultado do Held-Karp para 20 vértices.                                   |
| **results_palmer.csv**          | Resultados do método Palmer para  até 150 vértices.                  |
| **results_palmer_20.csv**       | Resultados do método Palmer para 20 vértices.                              |
| **results_vacul.csv**           | Resultados do método VaCul para  até 150 vértices.                   |
| **results_vacul_20.csv**        | Resultados do método VaCul para 20 vértices.                               |

### Arquivos de figura (.png)
| Arquivo                      | Descrição                                                        |
| ---------------------------- | ---------------------------------------------------------------- |
| **graficos_20.png**          | Gráficos gerados com os dados de 20 nós, comparando os métodos. |
| **graficos_150.png**         | Gráficos gerados com os dados de 150 vértices.                            |
| **grafico_palmer_unico.png** | Gráfico isolado para o método Palmer.                            |
