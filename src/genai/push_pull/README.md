# Modelo Push-Pull: agentes reativos e proativos

Este diretório ilustra a seção **"Modelo Push-Pull: agentes reativos e proativos"** das notas de aula 08.

O exemplo dá continuidade ao domínio de vendas usado em `src/genai/mcp`: há dados de faturamento, uma consulta de vendas, verificação de estoque e notificações. A diferença aqui não está na ferramenta usada, mas em **quem inicia a execução**:

- **Push**: um monitor agendado detecta uma queda de vendas sem esperar uma pergunta.
- **Pull**: o analista lê o alerta e faz perguntas para investigar.
- **Push final**: o sistema cria um pedido emergencial e notifica o responsável.

## Estrutura

- `sales_data.py`
  Dados simulados de vendas, estoque e histórico de incidentes.

- `tools.py`
  Ferramentas atômicas do domínio: `query_sales`, `check_stock`, `query_history`, `send_alert`, `create_restock_order` e `notify_manager`.

- `llm_support.py`
  Chamadas à API da OpenAI usadas para planejar ações, redigir alertas e sintetizar respostas a partir das observações das ferramentas.

- `agents.py`
  Dois modos de iniciação sobre o mesmo domínio:
  - `PushSalesMonitor`, proativo, disparado por horário/evento;
  - `PullInvestigationAgent`, reativo, disparado por perguntas do usuário.

- `demo.py`
  Script executável que percorre o ciclo:
  detecção -> investigação -> ação.
  Ao final, grava um log didático das chamadas ao LLM em `push_pull_demo_log.md`.

## Ordem de leitura sugerida

1. `sales_data.py`
2. `tools.py`
3. `llm_support.py`
4. `agents.py`
5. `demo.py`

## Como isso mudaria em produção

O exemplo mantém tudo em arquivos Python para que o fluxo fique fácil de ler. Em um sistema real, cada componente teria uma implementação diferente:

- `sales_data.py`
  Faz as vezes de uma fonte de dados persistente. Em produção, esses dados viriam de um banco relacional, data warehouse, lakehouse, sistema transacional ou API interna. As listas em memória seriam substituídas por tabelas como `vendas`, `estoque`, `produtos`, `incidentes` e `alertas`.

- `tools.py`
  Simula ferramentas de acesso ao domínio. Em produção, essas funções encapsulariam consultas SQL, chamadas HTTP, clientes de mensageria, SDKs internos ou tools expostas via MCP. Por exemplo, `query_sales` faria uma consulta parametrizada ao banco, `check_stock` consultaria o ERP ou serviço de estoque, e `send_alert` chamaria Slack, e-mail, Teams, PagerDuty ou outro canal.

- `llm_support.py`
  Simula a camada de raciocínio baseada em LLM. Em produção, essa camada teria prompts versionados, avaliação, limites de custo, telemetria, fallback, controle de modelos e validação mais rigorosa das saídas estruturadas. As chamadas ao LLM não acessam dados diretamente; elas recebem observações vindas das ferramentas e produzem planos ou textos.

- `agents.py`
  Contém a lógica de decisão do exemplo. Em produção, essa camada poderia ser implementada como um grafo LangGraph, um worker assíncrono, um serviço com estado, uma skill composta ou um host que combina LLM, ferramentas, memória e guardrails. O ponto importante é que o Push e o Pull podem usar as mesmas ferramentas, mas são iniciados por gatilhos diferentes.
  No exemplo, o agente Pull também valida e normaliza o plano produzido pelo LLM antes de executar a ferramenta. Esse detalhe é deliberado: em sistemas reais, o LLM não deve ter a última palavra sobre argumentos sensíveis ou contratos de ferramentas.

- `demo.py`
  Representa a orquestração. Em produção, a fase Push seria acionada por agenda, fila, webhook, stream de eventos ou monitoramento periódico. A fase Pull seria acionada por uma interface conversacional, API, dashboard ou integração com outro sistema. A ação final poderia exigir aprovação humana antes de criar pedidos, notificar gestores ou registrar incidentes.

Essa separação evita confundir o exemplo com uma arquitetura completa. O objetivo do código é mostrar o papel de cada componente no ciclo Push-Pull; a infraestrutura real entra depois, substituindo as simulações sem mudar o conceito central.

## Como rodar

O exemplo usa a SDK oficial da OpenAI, seguindo o mesmo padrão do exemplo em `src/genai/mcp`. Para executar o fluxo completo, você precisa de:

- `OPENAI_API_KEY` válida em `src/genai/.env` ou no ambiente;
- conectividade de rede com a API da OpenAI;
- dependências instaladas a partir do `requirements.txt`.

```bash
python3 src/genai/push_pull/demo.py
```

Se a chamada à API falhar, o script encerra com uma mensagem didática indicando que o problema provável está na chave, na rede ou no formato da resposta.

## O que observar

Na fase Push, o agente parte de um gatilho de monitoramento:

1. consulta vendas recentes;
2. compara com o limiar de queda;
3. verifica estoque para enriquecer o alerta;
4. usa o LLM para redigir e justificar o alerta;
5. envia uma notificação acionável.

Na fase Pull, o agente não age sozinho. Ele espera as perguntas do analista, usa o LLM para escolher a ferramenta adequada, valida o plano, executa a ferramenta e usa novamente o LLM para sintetizar a resposta. O pequeno estado de sessão permite encadear a investigação.

Na fase final, o sistema volta a agir em modo Push para registrar uma resposta coordenada. A mensagem ao gerente também é redigida pelo LLM a partir do diagnóstico estruturado.

O arquivo `push_pull_demo_log.md` permite inspecionar:

- os prompts enviados ao LLM;
- os planos estruturados produzidos;
- as respostas usadas pelo agente Push;
- as respostas usadas pelo agente Pull.

Esse exemplo mostra que Push-Pull não é uma topologia multi-agente. É uma decisão arquitetural sobre o **gatilho de execução**. O mesmo domínio e as mesmas ferramentas podem ser usados tanto em modo reativo quanto em modo proativo.
