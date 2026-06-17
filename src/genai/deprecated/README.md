# Projeto com LangChain, LLMs e Visualização

Este projeto integra ferramentas de IA generativa (como LangChain, OpenAI, Ollama) com bibliotecas de visualização, machine learning e deep learning. Também inclui uma interface interativa com o **Chainlit**.

---

## Requisitos

Certifique-se de ter o Python 3.10+ instalado (recomenda-se o uso de ambientes virtuais como `venv` ou `conda`).

---

## Instalação

Instale as dependências com:

```bash
pip install -r requirements.txt
````

---

## Usando o Chainlit

O **Chainlit** permite criar uma interface de chat para interagir com seu agente baseado em LLM.

---

## Executando o app Chainlit: Comparador de Receitas

Este projeto utiliza **LangGraph** e **Chainlit** para comparar duas receitas culinárias com base em arquivos de texto.

### Estrutura esperada

Certifique-se de ter os arquivos de receita no diretório `./data/`:

```
data/
├── recipe1.txt
└── recipe2.txt
```

Cada arquivo deve conter uma receita completa em formato texto. Esses conteúdos serão usados para compor o prompt do agente.

### Como rodar

Para iniciar a interface de comparação:

```bash
chainlit run app_recipes.py
```

A aplicação abrirá no navegador, permitindo que o usuário interaja com um agente capaz de:

* Identificar similaridades e diferenças entre as duas receitas;
* Responder a perguntas específicas com base no conteúdo das receitas;
* Informar quando os dados não são suficientes para responder adequadamente.

> ⚠️ Se os arquivos `recipe1.txt` ou `recipe2.txt` não forem encontrados, o sistema será encerrado com uma mensagem de erro.

### 🛠️ Dicas de uso

Você pode iniciar a conversa com perguntas como:

* "Quais ingredientes as duas receitas têm em comum?"
* "Qual receita leva mais tempo para preparar?"
* "Qual delas é mais saudável?"

---

## 📚 Estrutura das Bibliotecas

* **LangChain**: construção de agentes, cadeias de raciocínio e integração com ferramentas.
* **LangGraph**: implementação de agentes estruturados com controle de fluxo.
* **Ollama**: execução de LLMs locais.
* **OpenAI**: uso de modelos da OpenAI via API.
* **Chainlit**: interface web interativa para chat com o agente.
* **Matplotlib / Seaborn / Pandas**: visualização e análise de dados.
* **PyTorch**: treinamento de modelos customizados.
* **Gymnasium**: ambientes de reforço.
* **spaCy / Transformers**: NLP.

---

## Organização do Projeto

Recomenda-se a seguinte estrutura de arquivos:

```
.
├── app_recipes.py           # Entrada principal da aplicação Chainlit
├── data/
│   ├── recipe1.txt          # Primeira receita
│   └── recipe2.txt          # Segunda receita
├── notebooks/               # Análises exploratórias e testes
├── requirements.txt
└── README.md
```
