from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.genai.get_llm import load_env


MODEL = "gpt-4o-mini"


def call_openai_json(
    stage: str,
    messages: list[dict[str, str]],
    log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Chama a API da OpenAI esperando uma resposta JSON.

    Esta função concentra a integração real com o LLM. As outras funções deste
    módulo montam prompts específicos e delegam a chamada para ela.

    Args:
        stage: Nome da etapa, usado no log para rastrear a execução.
        messages: Lista de mensagens no formato da API de chat da OpenAI.
        log: Lista opcional que recebe prompts, respostas e erros para inspeção.

    Returns:
        Dicionário produzido a partir do JSON retornado pelo modelo.

    Raises:
        RuntimeError: Se a chamada falhar, a resposta vier vazia ou o conteúdo
            não puder ser interpretado como JSON.
    """
    try:
        load_env()
        client = OpenAI()
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        # print(f"LLM response for stage {stage}: {content}")
        
        if content is None:
            raise ValueError("A API retornou uma resposta vazia.")

        parsed = json.loads(content)
        if log is not None:
            log.append(
                {
                    "stage": stage,
                    "model": MODEL,
                    "messages": messages,
                    "response": parsed,
                }
            )
        return parsed
    except Exception as exc:
        if log is not None:
            log.append(
                {
                    "stage": stage,
                    "model": MODEL,
                    "messages": messages,
                    "error": str(exc),
                }
            )
        raise RuntimeError(
            "Falha ao chamar a OpenAI. Verifique a conectividade de rede, "
            "a OPENAI_API_KEY e o formato da resposta."
        ) from exc


def draft_push_alert(
    sales: dict[str, Any],
    stock: dict[str, Any],
    drop: float,
    threshold: float,
    log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pede ao LLM que redija e justifique um alerta Push.

    O agente já observou vendas e estoque por meio de ferramentas. O LLM recebe
    essas observações e decide como comunicar o alerta, sem acessar dados por
    conta própria.

    Args:
        sales: Observação retornada por `query_sales`.
        stock: Observação retornada por `check_stock`.
        drop: Queda relativa observada, em escala decimal. Exemplo: `0.47`.
        threshold: Limiar de alerta, também em escala decimal.
        log: Lista opcional para registrar prompt e resposta.

    Returns:
        Dicionário com `should_alert`, `severity`, `reasoning` e `message`.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um agente Push de monitoramento comercial. "
                "Receba observações de ferramentas e produza um JSON com as chaves "
                "'should_alert', 'severity', 'reasoning' e 'message'. "
                "'should_alert' deve ser booleano. 'severity' deve ser uma de: "
                "baixa, media, alta. A mensagem deve ser curta, acionável e em português."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "sales_observation": sales,
                    "stock_observation": stock,
                    "drop_percent": round(drop * 100, 2),
                    "threshold_percent": round(threshold * 100, 2),
                },
                ensure_ascii=False,
            ),
        },
    ]
    result = call_openai_json("draft_push_alert", messages, log)
    validate_required_keys(
        result,
        {
            "should_alert": bool,
            "severity": str,
            "reasoning": str,
            "message": str,
        },
    )
    return result


def plan_pull_tool_call(
    question: str,
    session_state: dict[str, Any],
    log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pede ao LLM que planeje uma chamada de ferramenta no modo Pull.

    Esta função representa a etapa em que o assistente interpreta a pergunta do
    analista e escolhe uma ferramenta. O plano ainda será validado pelo host em
    `agents.py` antes de qualquer ferramenta ser executada.

    Args:
        question: Pergunta feita pelo analista.
        session_state: Estado acumulado da investigação Pull.
        log: Lista opcional para registrar prompt e resposta.

    Returns:
        Dicionário com `tool`, `arguments` e `reasoning`.

    Raises:
        ValueError: Se o LLM escolher uma ferramenta fora da lista permitida ou
            retornar uma estrutura inválida.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um agente de investigação comercial. "
                "Escolha exatamente uma ferramenta para responder à pergunta do usuário. "
                "Responda apenas em JSON com as chaves 'tool', 'arguments' e 'reasoning'. "
                "Ferramentas disponíveis: "
                "query_sales(product: string|null, last_hours: integer|null, compare_all: boolean), "
                "query_history(product: string, cause: string), clarify(). "
                "Use product='X' quando a pergunta se referir ao alerta atual."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "session_state": session_state,
                },
                ensure_ascii=False,
            ),
        },
    ]
    result = call_openai_json("plan_pull_tool_call", messages, log)
    validate_required_keys(
        result,
        {
            "tool": str,
            "arguments": dict,
            "reasoning": str,
        },
    )
    if result["tool"] not in {"query_sales", "query_history", "clarify"}:
        raise ValueError(f"Ferramenta não permitida: {result['tool']}")
    return result


def draft_pull_answer(
    question: str,
    tool_plan: dict[str, Any],
    observation: dict[str, Any],
    session_state: dict[str, Any],
    log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pede ao LLM que sintetize a resposta ao analista.

    Nesta etapa, a ferramenta já foi executada. O LLM recebe a pergunta, o plano,
    a observação da ferramenta e o estado de sessão para produzir uma resposta em
    linguagem natural.

    Args:
        question: Pergunta original feita pelo analista.
        tool_plan: Plano de ferramenta validado pelo host.
        observation: Resultado retornado pela ferramenta executada.
        session_state: Estado acumulado da investigação Pull.
        log: Lista opcional para registrar prompt e resposta.

    Returns:
        Dicionário com a chave `answer`. Se o modelo devolver um objeto em vez
        de string, o valor é serializado para manter o fluxo didático executável.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente de análise comercial respondendo a uma pergunta do analista. "
                "Responda em português, de forma objetiva, usando apenas a observação "
                "da ferramenta e o estado de sessão. Responda apenas em JSON com a chave "
                "'answer'. O valor de 'answer' deve ser uma string em linguagem natural, "
                "nunca um objeto, lista ou tabela JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "tool_plan": tool_plan,
                    "observation": observation,
                    "session_state": session_state,
                },
                ensure_ascii=False,
            ),
        },
    ]
    result = call_openai_json("draft_pull_answer", messages, log)
    if "answer" not in result:
        raise ValueError("Resposta do LLM sem a chave obrigatória: answer")
    if not isinstance(result["answer"], str):
        result["answer"] = json.dumps(result["answer"], ensure_ascii=False)
    return result


def draft_manager_notification(
    diagnosis: dict[str, Any],
    log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pede ao LLM que redija a notificação final ao gestor.

    Esta função pertence à fase de ação coordenada. Ela transforma um diagnóstico
    estruturado em uma mensagem curta, clara e orientada à ação.

    Args:
        diagnosis: Dados estruturados sobre pedido, causa provável e impacto.
        log: Lista opcional para registrar prompt e resposta.

    Returns:
        Dicionário com a chave `message`. Se o modelo devolver outro tipo de
        valor, ele é serializado como JSON para preservar a execução.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Você redige notificações executivas para uma equipe comercial. "
                "Responda apenas em JSON com a chave 'message'. A mensagem deve ser "
                "curta, clara e orientada à ação."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(diagnosis, ensure_ascii=False),
        },
    ]
    result = call_openai_json("draft_manager_notification", messages, log)
    if "message" not in result:
        raise ValueError("Resposta do LLM sem a chave obrigatória: message")
    if not isinstance(result["message"], str):
        result["message"] = json.dumps(result["message"], ensure_ascii=False)
    return result


def validate_required_keys(result: dict[str, Any], schema: dict[str, type]) -> None:
    """Valida se a resposta estruturada do LLM segue o contrato esperado.

    Args:
        result: Dicionário retornado pelo LLM.
        schema: Mapeamento entre nome da chave e tipo esperado.

    Raises:
        ValueError: Se uma chave obrigatória estiver ausente ou vier com tipo
            diferente do esperado.
    """
    for key, expected_type in schema.items():
        if key not in result:
            raise ValueError(f"Resposta do LLM sem a chave obrigatória: {key}")
        if not isinstance(result[key], expected_type):
            raise ValueError(
                f"Chave {key!r} deveria ter tipo {expected_type.__name__}, "
                f"mas veio como {type(result[key]).__name__}."
            )
