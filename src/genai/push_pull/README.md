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
  - `ProductMonitorConfig`, configuração por produto com limiar, janela e destinatário;
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
  O agente Push monitora uma política de produtos (`ProductMonitorConfig`) em vez de um único produto fixo. Em produção, essa política viria de banco, API ou serviço de configuração.
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

## Mapeamento para as fases das notas

O exemplo resolvido das notas descreve três fases: detecção, investigação e ação coordenada. No código, elas aparecem assim:

- **Fase 1: detecção com agente Push**
  Implementada em `PushSalesMonitor.run_scheduled_check`, no arquivo `agents.py`.
  Essa função representa o processo agendado das notas. Ela percorre as configurações retornadas por `default_monitor_configs`, também em `agents.py`. Para cada produto configurado:
  1. chama `query_sales`, de `tools.py`, para observar as vendas recentes;
  2. compara a média observada com a média histórica definida na configuração do produto;
  3. se a queda passar do limiar, chama `check_stock`, de `tools.py`, para investigar uma causa provável;
  4. chama `draft_push_alert`, de `llm_support.py`, para redigir e justificar o alerta;
  5. chama `send_alert`, de `tools.py`, para enviar a notificação.

- **Fase 2: investigação com agente Pull**
  Implementada em `PullInvestigationAgent.answer`, no arquivo `agents.py`.
  Essa função representa o assistente usado pelo analista depois de ler o alerta. Cada pergunta do usuário percorre este fluxo:
  1. `plan_pull_tool_call`, em `llm_support.py`, pede ao LLM um plano de ferramenta;
  2. `normalize_pull_tool_plan`, em `agents.py`, valida e ajusta o plano ao contrato das ferramentas;
  3. `query_sales` ou `query_history`, em `tools.py`, executa a consulta;
  4. `draft_pull_answer`, em `llm_support.py`, sintetiza a resposta para o analista;
  5. `session_state`, em `PullInvestigationAgent`, preserva o contexto da investigação.

- **Fase 3: ação coordenada em modo Push**
  Implementada em `coordinate_final_action`, no arquivo `agents.py`.
  Essa função representa o fechamento operacional do ciclo. Ela:
  1. chama `create_restock_order`, de `tools.py`, para criar o pedido emergencial;
  2. chama `draft_manager_notification`, de `llm_support.py`, para redigir a mensagem ao gestor;
  3. chama `notify_manager`, de `tools.py`, para registrar a notificação.

O arquivo `demo.py` apenas encadeia essas três fases em uma execução única: primeiro instancia `PushSalesMonitor`, depois envia perguntas para `PullInvestigationAgent` e, por fim, chama `coordinate_final_action`.

## Rótulos do trace

O trace impresso pelo exemplo é um trace de execução sistêmica, não uma transcrição ReAct estrita. Por isso, os rótulos evitam sugerir que todo passo é raciocínio interno do LLM.

| Rótulo | Significado neste exemplo |
|---|---|
| `Trigger` | Algo iniciou o agente Push, como agenda, evento ou monitoramento periódico. |
| `Check` | O host selecionou um item da política de monitoramento para avaliar. |
| `Decision` | Regra determinística do host ou do agente, sem chamada ao LLM. |
| `ToolExecution` | O host Python executou uma ferramenta, como `query_sales`, `check_stock` ou `send_alert`. |
| `ToolOutput` | Resultado retornado pela ferramenta executada. |
| `LLM` | Chamada real ao modelo para planejar, justificar, sintetizar ou redigir. |

`ToolExecution` e `ToolOutput` são próximos das ideias de `Action` e `Observation` em ReAct, mas aqui eles descrevem chamadas controladas pelo código Python. Eles não significam que o LLM emitiu uma tool call nativa. O rótulo `LLM` marca inferência do modelo, não um `Thought` exposto.

## Intenção, validação e execução

Na fase Pull, o analista escreve em linguagem natural. O harness não sabe sozinho qual ferramenta executar. Ele usa o LLM para transformar a pergunta em uma intenção estruturada de ferramenta.

Exemplo de pergunta do analista:

```text
Mostre as vendas do produto X por hora nos últimos 7 dias.
```

`PullInvestigationAgent.answer`, em `agents.py`, chama `plan_pull_tool_call`, em `llm_support.py`. O LLM recebe a pergunta, o estado da sessão e a lista de ferramentas disponíveis. Ele deve devolver um JSON como:

```json
{
  "tool": "query_sales",
  "arguments": {
    "product": "X",
    "last_hours": 168,
    "compare_all": false
  },
  "reasoning": "A pergunta solicita vendas horárias do produto X nos últimos 7 dias."
}
```

Esse JSON ainda não executa nada. Ele é apenas uma intenção de ferramenta. Em seguida, `normalize_pull_tool_plan`, em `agents.py`, valida e ajusta o plano ao contrato do exemplo. Como a base didática não possui 168 horas reais, o host normaliza o plano para a janela simulada:

```json
{
  "tool": "query_sales",
  "arguments": {
    "product": "X",
    "last_hours": null,
    "compare_all": false
  },
  "reasoning": "Plano ajustado pelo host: a base didática possui a janela horária simulada do incidente."
}
```

Só depois disso o harness executa a ferramenta:

```python
query_sales(product="X", last_hours=None, compare_all=False)
```

O resultado da ferramenta vira uma observação estruturada:

```json
{
  "produto": "X",
  "media_historica_hora_fmt": "R$ 12.000,00",
  "media_observada_fmt": "R$ 6.720,00",
  "vendas": [
    {
      "hora": "2024-11-18 00:00",
      "faturamento_fmt": "R$ 6.120,00",
      "regiao": "Sul"
    }
  ]
}
```

Por fim, `draft_pull_answer`, em `llm_support.py`, recebe a pergunta original, o plano validado, a observação da ferramenta e o estado da sessão. O LLM então produz a resposta em linguagem natural para o analista.

Assim, a sequência correta é:

1. pergunta humana;
2. intenção de ferramenta produzida pelo LLM;
3. validação e normalização pelo host;
4. execução da ferramenta pelo host;
5. síntese da resposta pelo LLM.

## O que observar

Na fase Push, o agente parte de um gatilho de monitoramento:

1. lê uma política de monitoramento com produtos, janelas e limiares;
2. consulta vendas recentes de cada produto;
3. compara com o limiar de queda específico daquele produto;
4. verifica estoque para enriquecer alertas relevantes;
5. usa o LLM para redigir e justificar o alerta;
6. envia uma notificação acionável.

Na fase Pull, o agente não age sozinho. Ele espera as perguntas do analista, usa o LLM para escolher a ferramenta adequada, valida o plano, executa a ferramenta e usa novamente o LLM para sintetizar a resposta. O pequeno estado de sessão permite encadear a investigação.

Na fase final, o sistema volta a agir em modo Push para registrar uma resposta coordenada. A mensagem ao gerente também é redigida pelo LLM a partir do diagnóstico estruturado.

O arquivo `push_pull_demo_log.md` permite inspecionar:

- os prompts enviados ao LLM;
- os planos estruturados produzidos;
- as respostas usadas pelo agente Push;
- as respostas usadas pelo agente Pull.

Esse exemplo mostra que Push-Pull não é uma topologia multi-agente. É uma decisão arquitetural sobre o **gatilho de execução**. O mesmo domínio e as mesmas ferramentas podem ser usados tanto em modo reativo quanto em modo proativo.
