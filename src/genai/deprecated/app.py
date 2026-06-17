import os
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document # Importar Document para criar documentos fictícios

# --- 1. Configurações Iniciais ---
# Nome do modelo LLM no Ollama que será usado (certifique-se de que está puxado)
LLM_MODEL = "gemma3:latest"

# Para embeddings, usaremos um modelo de embedding dedicado
EMBEDDING_MODEL = "nomic-embed-text"

# --- 2. Função de Inicialização do Chat (Decorador Chainlit) ---
# Esta função é executada uma vez quando um novo chat é iniciado no Chainlit.
@cl.on_chat_start
async def start():
    # Mensagem de boas-vindas para o usuário
    await cl.Message(
        content=f"Olá! Eu sou um assistente especializado na Norma 175 e Ofícios CVM. "
                f"Pergunte-me algo sobre esses tópicos. (Usando LLM: {LLM_MODEL})"
    ).send()

    # --- 3. Carregar e Processar Documentos Fictícios (para Demonstração) ---
    # Em um cenário real, você carregaria PDFs/HTMLs de uma pasta 'docs'.
    # Aqui, criamos alguns documentos fictícios para que o RAG tenha algo para recuperar.
    print("Criando documentos fictícios para demonstração...")
    documents = [
        Document(
            page_content="A Norma 175 da CVM regula a constituição, o funcionamento e a administração "
                          "dos fundos de investimento no Brasil. Ela busca modernizar e consolidar "
                          "as regras existentes, trazendo maior flexibilidade e segurança jurídica "
                          "para o mercado de capitais. Um dos principais pontos é a permissão para "
                          "fundos de investimento em criptoativos, sob certas condições e com alertas de risco.",
            metadata={"source": "Norma 175 - Art. 1", "page": 1}
        ),
        Document(
            page_content="O Ofício Circular CVM/SNC/Nº 01/2023 esclarece dúvidas sobre a aplicação "
                          "da Norma 175, especialmente no que tange à segregação de ativos e "
                          "responsabilidades dos administradores de fundos. Ele reforça a importância "
                          "da diligência na avaliação de prestadores de serviços e na gestão de riscos "
                          "operacionais e de custódia, visando proteger os investidores.",
            metadata={"source": "Ofício Circular CVM/SNC/Nº 01/2023", "page": 3}
        ),
        Document(
            page_content="A Norma 175 permite a criação de diferentes classes de cotas para um mesmo fundo, "
                          "o que oferece maior flexibilidade na captação e na oferta de produtos. "
                          "Cada classe pode ter características distintas, como prazos de resgate, "
                          "taxas de administração e público-alvo, desde que devidamente detalhado "
                          "no regulamento do fundo e aprovado pela CVM.",
            metadata={"source": "Norma 175 - Art. 15", "page": 5}
        ),
        Document(
            page_content="Em relação à publicidade dos fundos, a Norma 175 exige que todo material "
                          "de divulgação seja claro, preciso e não induza o investidor a erro. "
                          "É obrigatória a inclusão de alertas sobre os riscos envolvidos, "
                          "especialmente para fundos com maior volatilidade ou exposição a ativos "
                          "considerados de risco elevado, como derivativos complexos ou criptoativos.",
            metadata={"source": "Norma 175 - Art. 50", "page": 10}
        )
    ]

    # Dividir os documentos em pedaços (chunks) menores
    print("Dividindo documentos em pedaços (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks gerados: {len(chunks)}")

    # --- 4. Criar Embeddings e Armazenar no Vector Store (ChromaDB) ---
    print("Gerando embeddings e construindo o banco de vetores Chroma (em memória)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # O ChromaDB será criado em memória para esta demonstração.
    # Para persistência, você usaria `persist_directory="./chroma_db"` e `Chroma(persist_directory=...)`
    vector_store = Chroma.from_documents(chunks, embeddings)
    print("Banco de vetores Chroma pronto.")

    # --- 5. Configurar o LLM Local (Gemma via Ollama) ---
    print(f"Inicializando o LLM '{LLM_MODEL}' via Ollama...")
    llm = Ollama(model=LLM_MODEL)
    print("LLM inicializado.")

    # --- 6. Construir a Cadeia RAG (Retrieval Augmented Generation) ---
    # System Prompt de Exemplo:
    system_prompt = (
        "Você é um assistente especializado e altamente preciso em regulamentação do mercado de capitais, "
        "com foco na Norma 175 da CVM e ofícios relacionados. "
        "Sua principal função é responder às perguntas dos usuários de forma clara, concisa e **estritamente baseada** "
        "nas informações fornecidas no contexto. "
        "Se a resposta não puder ser encontrada nas informações do contexto, diga educadamente que não possui essa informação. "
        "Mantenha um tom profissional e objetivo."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Contexto: {context}\n\nPergunta: {input}")
    ])

    # Cadeia para combinar os documentos recuperados com a pergunta
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Criar o retriever para buscar os documentos no banco de vetores
    # search_kwargs={"k": 3} busca os 3 chunks mais relevantes
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Criar a cadeia de recuperação completa
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Armazenar a cadeia na sessão do usuário Chainlit para ser usada em cada mensagem
    cl.user_session.set("retrieval_chain", retrieval_chain)
    print("Cadeia RAG configurada e pronta.")

# --- 7. Função de Mensagem do Chat (Decorador Chainlit) ---
# Esta função é executada toda vez que o usuário envia uma mensagem.
@cl.on_message
async def main(message: cl.Message):
    # Recuperar a cadeia RAG da sessão do usuário
    retrieval_chain = cl.user_session.get("retrieval_chain") # type: RetrievalChain

    # Exibir um indicador de carregamento enquanto a resposta é gerada
    msg = cl.Message(content="")
    await msg.send()

    # Invocar a cadeia RAG com o conteúdo da mensagem do usuário
    response = await retrieval_chain.ainvoke({"input": message.content})

    # A resposta da cadeia RAG tem a chave 'answer'
    final_answer = response["answer"]

    # Atualizar a mensagem com a resposta final
    await msg.stream_token(final_answer) # Envia a resposta token por token para uma experiência mais fluida

    # Opcional: Mostrar os documentos fonte que foram usados para gerar a resposta
    if "context" in response:
        sources = response["context"]
        if sources:
            source_elements = []
            for source_idx, source in enumerate(sources):
                # Formata a fonte para exibição
                source_name = source.metadata.get("source", f"Fonte Desconhecida {source_idx+1}")
                page_info = source.metadata.get("page", "N/A")
                source_elements.append(
                    await cl.Element(
                        name=f"Fonte {source_idx+1}: {source_name} (Pág. {page_info})",
                        content=source.page_content,
                        display="inline"
                    )
                )
            await cl.Message(
                content="**Fontes utilizadas:**",
                elements=source_elements
            ).send()
