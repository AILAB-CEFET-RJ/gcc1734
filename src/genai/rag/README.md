# RAG com LangChain e Chroma

Este diretório organiza o material didático de **Retrieval-Augmented Generation (RAG)** em módulos Python separados.

## Estrutura

- `sample_corpus.py`
  Cria o documento de exemplo usado no estudo de caso.

- `llm_config.py`
  Carregamento de `.env` e inicialização do LLM.

- `loaders.py`
  Carga de documentos.

- `chunking.py`
  Estratégias de divisão em chunks.

- `indexing.py`
  Embeddings e indexação no Chroma.

- `retrieval.py`
  Construção do retriever e recuperação de contexto.

- `generation.py`
  Prompt final e geração da resposta.

- `pipeline_demo.py`
  Script executável que percorre o fluxo completo:
  indexação → retrieval → generation.

## Ordem de leitura sugerida

1. `sample_corpus.py`
2. `loaders.py`
3. `chunking.py`
4. `indexing.py`
5. `retrieval.py`
6. `generation.py`
7. `pipeline_demo.py`

## Como rodar

Use as dependências de [requirements.txt](/Users/ebezerra/ailab/gcc1734/src/genai/requirements.txt).

Para executar o estudo de caso completo:

```bash
python src/genai/rag/pipeline_demo.py
```

## Observação didática

RAG fica mais claro quando as responsabilidades aparecem separadas:

- carregamento;
- chunking;
- embeddings e indexação;
- retrieval;
- geração.

Se você quiser manter um notebook companion, o ideal é que ele apenas importe esses módulos e use pequenas células para inspecionar:

- os chunks produzidos;
- os documentos recuperados;
- o prompt final;
- as respostas geradas.
