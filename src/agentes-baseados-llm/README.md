# Exemplo: Guardrails + LangGraph

Este diretório contém um exemplo autocontido para a aula de **Agentes baseados em LLMs**. A ideia é mostrar como encaixar **guardrails de entrada e de saída** dentro de um fluxo com LangGraph.

Esse exemplo:

- usa validadores customizados locais, em vez de depender do Guardrails Hub;
- usa a API atual do Guardrails (`Guard.use(...)`), em vez de `Guard.use_many(...)`;
- inclui um backend `mock`, para demonstração previsível e reproduzível;
- inclui um backend `ollama`, caso você queira conectar o grafo a um LLM local;
- realiza orquestração com LangGraph.

Isso conversa diretamente com as seções das notas sobre:
- memória conversacional;
- confiabilidade e segurança;
- guardrails como limites operacionais.

## O que o exemplo mostra

O grafo tem três etapas:

1. `input_guardrails`
   Valida a mensagem do usuário antes de chamar o modelo.

2. `model`
   Gera uma resposta com um LLM real (`ollama`) ou com um modelo simulado (`mock`).

3. `output_guardrails`
   Valida ou corrige a saída antes de devolvê-la ao usuário.

As políticas implementadas são:

- bloquear pedidos com cara de prompt injection / prompt leakage;
- bloquear pedidos com conteúdo claramente perigoso;
- mascarar e-mail e telefone caso aparecam na resposta do agente.

## Arquivos

| Arquivo | Papel |
| ------- | ----- |
| `guardrails_langgraph_demo.py` | Script principal com o grafo, os validadores e a interface de linha de comando. |
| `requirements-guardrails-langgraph.txt` | Dependências mínimas para este exemplo. |

## Instalação

Crie ou ative um ambiente virtual e instale as dependências do exemplo:

```bash
pip install -r src/agentes-baseados-llm/requirements-guardrails-langgraph.txt
```

Observação importante:
em **19 de maio de 2026**, o pacote `guardrails-ai` aparece no PyPI oficial como **quarantined**. Nessa condição, o próprio PyPI informa que o projeto não pode ser instalado por clientes `pip`. Por isso, este exemplo instala o Guardrails diretamente do repositório oficial no GitHub, com a versão fixada em `v0.9.3`.

Se você quiser usar o backend `ollama`, também precisa ter o servidor do Ollama em execução e um modelo baixado, por exemplo:

```bash
ollama pull llama3.1:8b
```

## Execução rápida

### 1. Exemplo de execução

```bash
python src/agentes-baseados-llm/guardrails_langgraph_demo.py --backend mock --demo
```

Esse modo executa três cenários prontos:

- uma pergunta normal sobre LangGraph;
- uma tentativa de jailbreak;
- uma pergunta que faz o modelo simulado produzir contato pessoal, para demonstrar a correção da saída.

### 2. Modo interativo com backend mock

```bash
python src/agentes-baseados-llm/guardrails_langgraph_demo.py --backend mock
```

Comandos úteis:

- `clear` limpa a conversa;
- `exit` ou `quit` encerra o programa.

### 3. Modo interativo com Ollama

```bash
python src/agentes-baseados-llm/guardrails_langgraph_demo.py --backend ollama --model llama3.1:8b
```

## Estrutura conceitual do exemplo

O fluxo implementa a ideia central de guardrails:

- **antes** do modelo: impedir que entradas inadequadas avancem pelo pipeline;
- **depois** do modelo: validar ou corrigir a resposta antes da entrega;
- **dentro de um grafo**: tornar o fluxo explícito, rastreável e extensível.

## Observações

- O backend `mock` produz comportamentos previsíveis.
- O backend `ollama` mostra que o mesmo grafo funciona com um LLM real.
- O exemplo não pretende ser um sistema de segurança de produção.
  Ele foi desenhado **para fins didáticos**.

## Referências

- Artigo-base: https://medium.com/@prajwalbm23/getting-started-with-guardrails-using-guardrails-ai-and-langgraph-560a5697a2ae
- Guardrails AI: https://guardrailsai.com/guardrails/docs/api_reference_markdown/guards
- Migração Guardrails 0.9: https://guardrailsai.com/guardrails/docs/migration-guides/0-9-migration
- Memória no LangGraph: https://docs.langchain.com/oss/python/langgraph/add-memory
