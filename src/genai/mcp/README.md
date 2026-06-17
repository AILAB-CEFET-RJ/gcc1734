# MCP com FastMCP

Este diretório ilustra a implementação de um agenteque usa **Model Context Protocol (MCP)**. A implementação está organizada em módulos Python separados.

## Estrutura

- `shared_data.py`
  Dados simulados e helpers comuns ao estudo de caso.

- `db_server.py`
  MCP Server de banco de dados. Expõe:
  - `query_database` como `tool`;
  - `resumir_faturamento` como `prompt`.

- `files_server.py`
  MCP Server de arquivos. Expõe:
  - `file:///relatorios/vendas/{period}/{region_slug}.pdf` como `resource`.

- `host_demo.py`
  Host, com as seguintes atribuições:
  - descobre capacidades;
  - monta contexto para o LLM;
  - chama `tools`, `prompts` e `resources`;
  - imprime um trace da execução.

## Ordem de leitura sugerida

1. `shared_data.py`
2. `db_server.py`
3. `files_server.py`
4. `host_demo.py`

Essa ordem acompanha a arquitetura da seção:

1. o que os servidores expõem;
2. como o host descobre capacidades;
3. como a chamada atravessa as camadas;
4. como o resultado volta ao host.

## Como rodar

Use as dependências de [requirements.txt](/Users/ebezerra/ailab/gcc1734/src/genai/requirements.txt).

Para executar o estudo de caso completo:

```bash
python src/genai/mcp/host_demo.py
```

Observação:

o `host_demo.py` usa a SDK oficial da OpenAI diretamente em duas etapas anteriores à chamada MCP:

1. para inferir a região mencionada na pergunta do usuário;
2. para planejar a chamada estruturada da tool `query_database`.

Essa escolha foi feita para deixar o fluxo mais fiel ao cenário das notas, sem puxar dependências desnecessárias do LangChain nessa etapa de interpretação e planejamento.

Portanto, para o fluxo completo funcionar, você precisa de:

- `OPENAI_API_KEY` válida em `src/genai/.env` ou no ambiente;
- conectividade de rede com a API da OpenAI;
- ambiente Python sem conflitos binários nas dependências.

Se essa etapa falhar, o script encerra com uma mensagem didática explicando o motivo provável, em vez de despejar um traceback longo.

## Log didático

Para não poluir a saída principal, o script imprime no terminal apenas:

- a pergunta do usuário;
- a resposta final.

Todo o restante fica registrado em:

`src/genai/mcp/host_demo_log.md`

Esse arquivo permite ao aluno inspecionar depois da execução:

- o trace completo do host;
- o prompt usado para inferir a região;
- a resposta do modelo para essa inferência;
- o prompt usado para planejar a chamada da tool;
- o JSON devolvido pelo modelo com `tool`, `arguments` e `reasoning`.
- os detalhes da leitura do `resource` em PDF.

O formato Markdown foi escolhido para permitir uma leitura guiada por seções, enquanto os blocos JSON continuam disponíveis dentro do log para ilustrar as mensagens estruturadas trocadas ao longo do fluxo.

Para inspecionar apenas um servidor MCP:

```bash
python src/genai/mcp/db_server.py
python src/genai/mcp/files_server.py
```

## Observação

Esta estrutura reflete de módulos torna visível a separação entre:

- host;
- MCP Client;
- MCP Servers;
- dados compartilhados.
