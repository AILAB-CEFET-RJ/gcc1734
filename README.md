# GCC1734 – Inteligência Artificial (CEFET/RJ)

Repositório utilizado na disciplina de IA.

---

## Organização principal

- `src/rl/` – agentes de Q-Learning (tabular, aproximação linear e rede neural com replay) e utilitários de ambientes Gymnasium.
- `src/llm/`, `src/multiagent/`, `src/genai/` – exemplos voltados para modelos de linguagem e agentes.
- `notebooks/` – materiais de apoio em Jupyter.
- `requirements.txt` – dependências gerais do repositório.

Cada subdiretório relevante contém um README específico com detalhes adicionais (por exemplo `src/rl/README-qlt.md`, `README-qll.md`, `README-qln.md`).

---

## Ambiente recomendado

1. **Criar o ambiente**
   ```bash
   conda create -n gcc1734 python=3.10
   conda activate gcc1734
   ```

   > Se preferir, use `python -m venv .venv && source .venv/bin/activate`.

2. **Instalar dependências**
   ```bash
   pip install --no-build-isolation -e .
   ```
   - Caso não queira a instalação editável, use `pip install -r requirements.txt`.
   - Adicione `--user` se encontrar problemas de permissão.

3. **Verificar**
   ```bash
   python -m rl.train_qlearning --help
   ```

---

## Exemplos rápidos

- **Treinar agente tabular Taxi-v3**
  ```bash
  python -m rl.train_qlearning --agent tabular --env_name Taxi-v3 --num_episodes 8000
  ```

- **Treinar agente linear com replay**
  ```bash
  python -m rl.train_qlearning --agent linear --env_name Taxi-v3 --num_episodes 5000 --plot
  ```

- **Executar notebooks**
  ```bash
  jupyter notebook
  ```

Arquivos gerados (modelos `.pkl`, curvas `.png`) ficam na raiz do projeto por padrão.

---

## Contribuindo

1. Crie uma branch para suas alterações.
2. Garanta que scripts/notebooks rodem antes do commit.
3. Abra um pull request descrevendo o que foi alterado.

Códigos e materiais podem ser reutilizados livremente para fins acadêmicos e de ensino.
