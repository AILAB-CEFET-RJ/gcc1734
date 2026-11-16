# GCC1734 — Companion 03: Alinhamento por preferências

## Conteúdo
- `03_alinhamento_preferencias.ipynb`: notebook com Bradley–Terry linear, DPO explícito, avaliação e experimento opcional de ruído nas preferências.
- `preferencias_sinteticas.csv`: 1.920 pares derivados do dataset SFT (1.536 treino / 384 validação), com critérios de correção e formato.

## Ordem de execução
1. Instale as dependências de `requirements.txt` (incluído no pacote completo).
2. Abra os notebooks **na pasta `llm_pretraining_companion`**; todos os caminhos são relativos a ela.
3. Execute `01_pre_treinamento.ipynb` e salve `artefatos_pretraining/modelo` e `artefatos_pretraining/tokenizador`.
4. Execute `02_ajuste_supervisionado.ipynb` e salve `artefatos_sft/modelo` e `artefatos_sft/tokenizador`.
5. Execute `03_alinhamento_preferencias.ipynb`; o checkpoint final será salvo em `artefatos_alinhamento`.

## Uso didático
O modelo é pequeno e o corpus é sintético. As preferências são geradas por regras, sem anotação humana. As métricas de pares não substituem avaliação de geração livre.

O experimento de Bradley–Terry usa características calculadas com base no gabarito; ele serve somente para ilustrar o aprendizado de escores, não como avaliador independente. O DPO é implementado explicitamente em PyTorch, mantendo uma política de referência congelada. A perda DPO usa somas de log-probabilidades sobre os tokens das respostas, incluindo EOS.

## Validação
Foram verificadas a estrutura JSON do notebook, a sintaxe de suas células Python, a separação dos datasets e a consistência dos arquivos. A execução numérica integral requer `transformers` e `tokenizers`, ausentes no ambiente de geração. Não há resultados de treinamento inventados.
