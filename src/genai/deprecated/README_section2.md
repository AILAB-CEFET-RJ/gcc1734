# LLM-Based Agents: Iterative Reasoning and Planning with LangChain and LangGraph

This notebook suite supports practical activities for Section 2 of the short course on LLM-based agents. It demonstrates how to build agents capable of iterative reasoning, tool use, memory, and planning using LangChain and LangGraph.

## Notebooks Overview

| Notebook                                   | Description                                              |
|--------------------------------------------|----------------------------------------------------------|
| 01_langchain_react_intro.ipynb             | ReAct-style agent with reasoning and tool use           |
| 02_langgraph_plan_and_act.ipynb            | Plan-and-Act architecture using LangGraph               |
| 03_langgraph_reflexion_extension.ipynb     | Self-correcting agent inspired by Reflexion             |
| 04_langgraph_with_memory.ipynb             | LangGraph agent with conversational memory integration  |

## Installation

Install the required packages in a new virtual environment:

```bash
pip install langchain langgraph openai duckduckgo-search
pip install langchain-community  # for ChatOllama
```

To use local models via Ollama:

```bash
pip install ollama
ollama run gemma:3b  # or other models such as mistral, llama2, etc.
```

## Model Configuration

Each notebook is parameterized to support two backends:

- `ChatOpenAI`: uses the OpenAI API (requires an API key)
- `ChatOllama`: uses a local model via the Ollama server

Set the desired backend at the top of each notebook:

```python
MODEL_BACKEND = "openai"  # or "ollama"
llm = get_llm(model_backend=MODEL_BACKEND)
```

By default:
- `openai` uses `"gpt-3.5-turbo"`
- `ollama` uses `"gemma:3b"`

Make sure the Ollama model is downloaded before running locally:
```bash
ollama run gemma:3b
```

## Execution

Launch Jupyter Notebook or JupyterLab and open the notebook of interest:

```bash
jupyter notebook
```

Each notebook contains a complete working example that runs on either OpenAI or Ollama, depending on configuration.

## Notes

- Notebooks using `ChatOpenAI` require a valid API key set via the `OPENAI_API_KEY` environment variable.
- LangGraph is under active development; ensure the latest version is installed.
