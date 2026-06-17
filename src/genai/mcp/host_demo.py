from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import sys
from typing import Any

from fastmcp import Client, FastMCP
from openai import OpenAI


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    sys.path.append(str(Path(__file__).resolve().parent))
    from src.genai.get_llm import load_env
    from db_server import db_server
    from files_server import files_server
    from shared_data import (
        DB_SCHEMA_METADATA,
        content_to_text,
        format_brl,
        format_int_pt_br,
        format_period_pt_br,
        pick_attr,
        pretty_json,
        region_to_slug,
        uri_template_to_regex,
    )
else:
    from src.genai.get_llm import load_env
    from .db_server import db_server
    from .files_server import files_server
    from .shared_data import (
        DB_SCHEMA_METADATA,
        content_to_text,
        format_brl,
        format_int_pt_br,
        format_period_pt_br,
        pick_attr,
        pretty_json,
        region_to_slug,
        uri_template_to_regex,
    )


SERVERS = {
    "db": db_server,
    "files": files_server,
}


def default_log_path() -> Path:
    return Path(__file__).resolve().parent / "host_demo_log.md"


def append_log(log: dict[str, Any], stage: str, prompt: Any, response: Any) -> None:
    log.setdefault("llm_calls", []).append(
        {
            "stage": stage,
            "prompt": prompt,
            "response": response,
        }
    )


def to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def render_markdown_log(log: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append("# Host Demo Log\n")
    parts.append("## Query\n")
    parts.append(f"{log.get('user_question', '')}\n")

    llm_calls = log.get("llm_calls", [])
    if llm_calls:
        parts.append("## LLM Calls\n")
        for i, call in enumerate(llm_calls, start=1):
            parts.append(f"### Call {i}: {call.get('stage', 'unknown')}\n")
            parts.append("Prompt:\n")
            parts.append("```json\n")
            parts.append(f"{to_pretty_json(call.get('prompt'))}\n")
            parts.append("```\n")
            parts.append("Response:\n")
            parts.append("```json\n")
            parts.append(f"{to_pretty_json(call.get('response'))}\n")
            parts.append("```\n")

    if "trace" in log:
        parts.append("## Host Trace\n")
        for i, step in enumerate(log["trace"]):
            parts.append(f"{i}. {step}\n")

    if "resource_read" in log:
        parts.append("## Resource Read\n")
        parts.append("```json\n")
        parts.append(f"{to_pretty_json(log['resource_read'])}\n")
        parts.append("```\n")

    return "\n".join(parts).strip() + "\n"


def write_log(log: dict[str, Any], path: Path | None = None) -> Path:
    target = path or default_log_path()
    target.write_text(render_markdown_log(log), encoding="utf-8")
    return target


def append_trace(log: dict[str, Any], trace: list[str]) -> None:
    log["trace"] = trace


def simulate_user_revenue_question(region: str) -> str:
    # Esta função representa o momento em que o humano formula a pergunta.
    # Em um sistema real, esse texto viria de chat, API ou interface gráfica.
    return f"Qual foi o faturamento total da regiao {region} no mes passado?"


def infer_region_from_question(user_question: str, log: dict[str, Any] | None = None) -> str:
    # Aqui usamos a API da OpenAI diretamente para transformar a pergunta
    # em um argumento estruturado simples, sem envolver toda a pilha do LangChain.
    try:
        load_env()
        client = OpenAI()
        messages = [
            {
                "role": "system",
                "content": (
                    "Extraia a regiao geografica mencionada na pergunta do usuario. "
                    "Responda com apenas uma das opcoes exatas: Sul, Sudeste, Nordeste. "
                    "Se nao houver regiao identificavel, responda apenas: DESCONHECIDA."
                ),
            },
            {
                "role": "user",
                "content": user_question,
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=messages,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("A API retornou uma resposta vazia.")
        region = content.strip()
        if log is not None:
            append_log(
                log,
                "infer_region_from_question",
                messages,
                {"content": region},
            )

        if region in {"Sul", "Sudeste", "Nordeste"}:
            return region

        raise ValueError(
            "O LLM nao retornou uma regiao valida para a pergunta do usuario."
        )
    except Exception as exc:
        if log is not None:
            append_log(
                log,
                "infer_region_from_question",
                {"user_question": user_question},
                {"error": str(exc)},
            )
        raise RuntimeError(
            "Falha ao inferir a regiao com a OpenAI. "
            "Verifique a conectividade de rede, a OPENAI_API_KEY e as dependencias do ambiente."
        ) from exc


def plan_database_tool_call(
    user_question: str,
    db_schema_metadata: dict[str, Any],
    log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Em um sistema mais realista, o LLM não recebe apenas a pergunta do usuário:
    # ele também recebe a descrição da tool e metadados do domínio para planejar a chamada.
    try:
        load_env()
        client = OpenAI()
        messages = [
            {
                "role": "system",
                "content": (
                    "Voce esta planejando uma chamada de tool MCP para um host.\n"
                    "Sua tarefa e produzir um objeto JSON com as chaves "
                    "'tool', 'arguments' e 'reasoning'.\n"
                    "A tool disponivel e 'query_database'.\n"
                    "Em 'arguments', inclua apenas a chave 'sql'.\n"
                    "O SQL deve ser somente leitura e compativel com o esquema informado.\n"
                    "Nao inclua markdown, cercas de codigo ou texto fora do JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Pergunta do usuario: {user_question}\n\n"
                    f"Esquema do banco: {json.dumps(db_schema_metadata, ensure_ascii=False)}\n\n"
                    "Produza a chamada de tool apropriada."
                ),
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("A API retornou uma resposta vazia.")

        plan = json.loads(content)
        if log is not None:
            append_log(
                log,
                "plan_database_tool_call",
                messages,
                plan,
            )
        if plan.get("tool") != "query_database":
            raise ValueError("O plano nao retornou a tool esperada.")
        if not isinstance(plan.get("arguments"), dict) or "sql" not in plan["arguments"]:
            raise ValueError("O plano nao retornou argumentos validos para query_database.")
        return plan
    except Exception as exc:
        if log is not None:
            append_log(
                log,
                "plan_database_tool_call",
                {
                    "user_question": user_question,
                    "db_schema_metadata": db_schema_metadata,
                },
                {"error": str(exc)},
            )
        raise RuntimeError(
            "Falha ao planejar a chamada da tool com a OpenAI. "
            "Verifique a conectividade de rede, a OPENAI_API_KEY e o formato da resposta."
        ) from exc


async def describe_server(server_name: str, server: FastMCP) -> dict[str, Any]:
    # O Client é a peça que "fala MCP" em nome do host.
    # Aqui ele consulta o catálogo publicado por um servidor.
    async with Client(server) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()

    return {
        "server": server_name,
        "tools": [
            {
                "name": pick_attr(tool, "name"),
                "description": pick_attr(tool, "description"),
            }
            for tool in tools
        ],
        "resources": [
            {
                "uri": pick_attr(resource, "uri"),
                "name": pick_attr(resource, "name"),
                "mime_type": pick_attr(resource, "mimeType", "mime_type"),
            }
            for resource in resources
        ],
        "resource_templates": [
            {
                "uri_template": pick_attr(template, "uriTemplate", "uri_template"),
                "name": pick_attr(template, "name"),
            }
            for template in templates
        ],
        "prompts": [
            {
                "name": pick_attr(prompt, "name"),
                "description": pick_attr(prompt, "description"),
            }
            for prompt in prompts
        ],
    }


class SalesAnalyticsHost:
    # O host é a aplicação-orquestradora completa:
    # ele descobre capacidades, monta contexto, decide o próximo passo
    # e usa o Client para conversar com servidores MCP.
    def __init__(self, servers: dict[str, FastMCP], db_schema_metadata: dict[str, Any]):
        self.servers = servers
        self.db_schema_metadata = db_schema_metadata

    async def describe_capabilities(self) -> list[dict[str, Any]]:
        catalog = []
        for server_name, server in self.servers.items():
            catalog.append(await describe_server(server_name, server))
        return catalog

    async def _find_tool_server(self, tool_name: str) -> str:
        for server_name, server in self.servers.items():
            async with Client(server) as client:
                tools = await client.list_tools()
                if any(pick_attr(tool, "name") == tool_name for tool in tools):
                    return server_name
        raise ValueError(f"Tool nao encontrada: {tool_name}")

    async def _find_prompt_server(self, prompt_name: str) -> str:
        for server_name, server in self.servers.items():
            async with Client(server) as client:
                prompts = await client.list_prompts()
                if any(pick_attr(prompt, "name") == prompt_name for prompt in prompts):
                    return server_name
        raise ValueError(f"Prompt nao encontrado: {prompt_name}")

    async def _find_resource_server(self, uri: str) -> str:
        for server_name, server in self.servers.items():
            async with Client(server) as client:
                resources = await client.list_resources()
                if any(pick_attr(resource, "uri") == uri for resource in resources):
                    return server_name

                templates = await client.list_resource_templates()
                for template in templates:
                    uri_template = pick_attr(template, "uriTemplate", "uri_template")
                    if uri_template and uri_template_to_regex(uri_template).match(uri):
                        return server_name

        raise ValueError(f"Resource nao encontrado: {uri}")

    async def build_llm_context(self, user_question: str) -> dict[str, Any]:
        return {
            "user_question": user_question,
            "catalog": await self.describe_capabilities(),
            "db_schema": self.db_schema_metadata,
            "host_note": (
                # Em MCP, o LLM não chama o servidor diretamente.
                # Ele raciocina a partir do contexto montado pelo host.
                "O LLM nao conversa diretamente com MCP Servers. "
                "Ele recebe este contexto montado pelo host."
            ),
        }

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], trace: list[str]) -> Any:
        server_name = await self._find_tool_server(tool_name)
        trace.append(
            f"Passo 2: o MCP Client roteou a tool '{tool_name}' para o servidor '{server_name}'."
        )

        # Esta é a chamada MCP propriamente dita: o host pede ao Client
        # que invoque uma tool remota (ou, neste demo, exposta em memória).
        async with Client(self.servers[server_name]) as client:
            result = await client.call_tool(tool_name, arguments)

        trace.append(
            f"Passo 3: o MCP Server '{server_name}' validou a chamada, executou a tool e devolveu um payload padronizado."
        )
        return result

    async def get_prompt(self, prompt_name: str, arguments: dict[str, Any], trace: list[str]) -> Any:
        server_name = await self._find_prompt_server(prompt_name)
        trace.append(
            f"MCP Client identificou que o prompt '{prompt_name}' pertence ao servidor '{server_name}'."
        )

        # Prompt em MCP não é resposta final; é um template reutilizável
        # que o host pode incorporar ao contexto do LLM.
        async with Client(self.servers[server_name]) as client:
            result = await client.get_prompt(prompt_name, arguments)

        trace.append(
            f"MCP Server '{server_name}' renderizou o prompt '{prompt_name}'."
        )
        return result

    async def read_resource(self, uri: str, trace: list[str]) -> list[Any]:
        server_name = await self._find_resource_server(uri)
        trace.append(
            f"MCP Client roteou o resource '{uri}' para o servidor '{server_name}'."
        )

        # Resource é acesso a dados identificados por URI.
        # Diferente de tool, aqui não pedimos "execute uma ação",
        # e sim "leia este dado".
        async with Client(self.servers[server_name]) as client:
            contents = await client.read_resource(uri)

        trace.append(
            f"MCP Server '{server_name}' leu o resource '{uri}' e devolveu seu conteudo."
        )
        return contents

    async def handle_revenue_question(self, user_question: str) -> dict[str, Any]:
        trace: list[str] = []
        llm_log: dict[str, Any] = {"user_question": user_question}
        region = infer_region_from_question(user_question, llm_log)

        trace.append(
            "Passo 0: o host consultou, por meio do MCP Client, o catalogo de capacidades publicado pelos servidores."
        )
        # O contexto combina: pergunta do usuário + catálogo MCP + metadados do domínio.
        # É isso que permite ao LLM propor uma chamada coerente.
        llm_context = await self.build_llm_context(user_question)
        trace.append(
            "O host montou o contexto do LLM com a pergunta do usuario, a descricao da tool, o esquema do banco e as restricoes de acesso."
        )

        tool_plan = plan_database_tool_call(user_question, self.db_schema_metadata, llm_log)
        sql = tool_plan["arguments"]["sql"]
        trace.append(
            "Passo 1: com base nesse contexto, o LLM no host formulou uma chamada estruturada para a tool query_database."
        )
        trace.append(
            f"Plano do LLM no host: tool={tool_plan['tool']} args={tool_plan['arguments']}"
        )

        tool_result = await self.call_tool("query_database", {"sql": sql}, trace)
        # No FastMCP, o resultado da tool volta encapsulado em um objeto de resposta.
        # Aqui extraímos apenas o dicionário útil para o restante do fluxo.
        data = tool_result.data
        trace.append(
            "Passo 4: o host recebeu o resultado da tool e o injetou como observacao no contexto."
        )

        prompt_result = await self.get_prompt(
            "resumir_faturamento",
            {
                "region": data["region"],
                "period": data["periodo"],
                "faturamento_total": data["faturamento_total"],
                "registros": data["registros"],
            },
            trace,
        )
        rendered_prompt = "\n".join(
            content_to_text(message.content) for message in prompt_result.messages
        )

        # A resposta final é gerada pelo host a partir da observação já obtida.
        # Neste demo, fazemos isso de forma determinística para deixar o fluxo explícito.
        period_label = format_period_pt_br(data["periodo"])
        revenue_label = format_brl(data["faturamento_total"])
        records_label = format_int_pt_br(data["registros"])
        final_answer = (
            f"O faturamento total da regiao {data['region']} no mes passado "
            f"({period_label}) foi de {revenue_label}, consolidado a partir de "
            f"{records_label} registros."
        )
        trace.append(
            "Passo 5: com o resultado no contexto, o host produziu a resposta final em linguagem natural."
        )
        append_trace(llm_log, trace)

        return {
            "llm_context": llm_context,
            "tool_plan": tool_plan,
            "sql": sql,
            "tool_result": data,
            "rendered_prompt": rendered_prompt,
            "final_answer": final_answer,
            "trace": trace,
            "llm_log": llm_log,
        }

    async def open_detailed_report(self, region: str, period: str) -> dict[str, Any]:
        trace: list[str] = []
        uri = f"file:///relatorios/vendas/{period}/{region_to_slug(region)}.pdf"

        trace.append("O host recebeu um novo pedido: ler o relatorio detalhado em PDF.")
        trace.append(
            "Agora nao se trata de uma tool; trata-se de um resource identificado por URI."
        )

        contents = await self.read_resource(uri, trace)
        item = contents[0]
        blob = pick_attr(item, "blob")
        mime_type = pick_attr(item, "mimeType", "mime_type")

        # Alguns clients materializam o blob como base64; outros, como bytes.
        # O tratamento abaixo cobre os dois casos sem complicar o exemplo.
        if isinstance(blob, str):
            blob_bytes = base64.b64decode(blob)
        else:
            blob_bytes = blob

        # Se o conteúdo começa com a assinatura de PDF, normalizamos o MIME
        # para que a saída didática fique consistente com o cenário das notas.
        if mime_type == "application/octet-stream" and blob_bytes.startswith(b"%PDF"):
            mime_type = "application/pdf"

        return {
            "uri": uri,
            "mime_type": mime_type,
            "bytes_len": len(blob_bytes),
            "bytes_preview_base64": base64.b64encode(blob_bytes[:48]).decode("ascii"),
            "trace": trace,
        }


async def main() -> None:
    host = SalesAnalyticsHost(SERVERS, DB_SCHEMA_METADATA)
    region = "Sul"
    log_path = default_log_path()

    print("\n=== Consulta principal ===")
    # Aqui simulamos explicitamente a fala do usuário antes de o host começar a orquestração.
    user_question = simulate_user_revenue_question(region)
    print("QUERY:")
    print(user_question)

    # O host recebe a pergunta já pronta e passa a coordenar o fluxo MCP.
    try:
        main_result = await host.handle_revenue_question(user_question)
    except RuntimeError as exc:
        print("Falha ao processar a pergunta com ajuda do LLM:")
        print(exc)
        print()
        print("Checklist rapido:")
        print("- OPENAI_API_KEY configurada em src/genai/.env ou no ambiente")
        print("- acesso de rede disponivel para a API da OpenAI")
        print("- dependencias do ambiente sem conflitos binarios")
        return

    written_log = write_log(main_result["llm_log"], log_path)

    print("\nANSWER:")
    print(main_result["final_answer"])
    print("\nLog detalhado:")
    print(written_log)

    # A leitura do resource continua fazendo parte do script,
    # mas seus detalhes agora ficam apenas no log para não poluir o stdout.
    report_result = await host.open_detailed_report(region, "2024-11")
    main_result["llm_log"]["resource_read"] = {
        "uri": report_result["uri"],
        "mime_type": report_result["mime_type"],
        "bytes_len": report_result["bytes_len"],
        "bytes_preview_base64": report_result["bytes_preview_base64"],
        "trace": report_result["trace"],
    }
    write_log(main_result["llm_log"], log_path)


if __name__ == "__main__":
    # FastMCP expõe operações assíncronas no Client.
    # Por isso, executamos o fluxo principal dentro
    # de um event loop.
    asyncio.run(main())
