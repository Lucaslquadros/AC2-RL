# Trabalho AC2 — Métodos de Monte Carlo (Aprendizado por Reforço, Ibmec)

**Entrega do relatório/apresentação:** 16/10/2026 · **Grupo:** _(até 3 alunos — preencher)_

Algoritmos estudados no ambiente **Blackjack-v1** (Gymnasium): First-Visit MC Prediction,
Monte Carlo ES (Exploring Starts) e On-Policy MC Control com ε-greedy.
Base: notebook do professor (`referencia/`), Sutton & Barto (2018), cap. 5.

## Estrutura

```
AC2/
├── Trabalho_MC__AC2_.pdf          enunciado
├── README.md
├── requirements.txt
├── referencia/
│   └── Aula_13_14_Monte_Carlo_RL_Didatico.ipynb   notebook do professor (original, não editar)
├── notebooks/
│   └── experimentos.ipynb         funções reaproveitadas + experimentos do grupo
├── resultados/                    CSVs e figuras (PNG) gerados pelo notebook, prontos para os slides
└── apresentacao/                  PPTX do trabalho
```

## Como executar (Windows / PowerShell)

```powershell
cd C:\Faculdade\2026_2\APRENDIZADO_POR_REFORCO\AC2
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab            # abra notebooks/experimentos.ipynb e rode as células em ordem
```

- **Teste rápido** (~10× menor, ~1–2 min): `$env:MC_RAPIDO="1"` antes de abrir o Jupyter, ou `MODO_RAPIDO = True` na célula de configuração.
- **Execução completa:** estimativa de 20–25 min. Todos os gráficos/tabelas vão para `resultados/`.

## O que o notebook `experimentos.ipynb` faz

| Seção | Conteúdo | Item da apresentação |
|---|---|---|
| 0 | Mudanças em relação ao notebook do professor e por quê | 1 |
| 2, 4, 5 | Código comentado: ambiente, First-Visit, ES, On-Policy ε-greedy | 1 |
| 3 | First-Visit com as políticas do grupo (desempenho, heatmaps de V, convergência) | 2 |
| 6 | Experimentos B (ES × On-Policy, ε), C (decaimento de ε), D (γ), E (viés das somas < 12) | 3 |
| 8 | Mapa código ↔ teoria e perguntas para a análise | 1 |

## A fazer pelo grupo

- [x] Definir **as duas (ou mais) políticas do grupo** (seção 3.1, dicionário `POLITICAS`):
      **"Imita o dealer (≥17)"** (copia a regra do próprio dealer) e **"Depende da carta do dealer"**
      (se a carta do dealer estiver entre 2 e 5, para cedo com soma ≥12; com qualquer outra carta,
      para com soma ≥17 — ideia baseada na observação dos heatmaps de política da seção 6).
- [ ] Rodar o notebook completo (com as políticas novas) e conferir `resultados/` — **os CSVs/PNGs
      atuais ainda são da versão anterior das políticas e precisam ser regenerados**
- [ ] Escrever as conclusões (seção 8.3) e a análise da implementação × teoria
- [ ] Montar o PPTX (`apresentacao/`) com os 3 itens do enunciado
- [ ] Combinar a divisão de tarefas e revisar antes de 16/10

## Observações importantes sobre o código original

1. **Somas < 12:** o ambiente tem 280 estados (somas 4–21), mas o ES do notebook original só sorteia 12–21 e avalia com STICK nos
   estados não aprendidos. Isso piora o ES de forma artificial. Corrigido aqui (seção 0 e Experimento E).
2. **First-Visit = Every-Visit no Blackjack:** nenhum estado se repete dentro de um episódio (verificado, seção 3.2).
3. Linhas alteradas em relação ao original estão marcadas com `# [ALTERADO]` ou `[NOVO]`.
