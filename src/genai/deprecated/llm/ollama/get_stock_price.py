import ollama  # Biblioteca para interação com modelos de linguagem
import yfinance as yf  # Biblioteca para obtenção de dados financeiros
from typing import Dict, Any, Callable  # Tipos auxiliares para anotações de tipo

def get_stock_price(symbol: str) -> float:
    """
    Obtém o preço atual de uma ação a partir do símbolo informado.

    Parâmetros:
        symbol (str): O símbolo da ação (e.g., 'AAPL', 'GOOGL').

    Retorno:
        float: O preço atual da ação.

    Exceções:
        Exception: Lançada se nenhum preço válido for encontrado.
    """
    ticker = yf.Ticker(symbol)  # Cria um objeto do Yahoo Finance para o símbolo fornecido
    
    # Lista de possíveis atributos onde o preço pode estar armazenado
    price_attrs = ['regularMarketPrice', 'currentPrice', 'price']
    
    # Percorre os atributos e retorna o primeiro valor válido encontrado
    for attr in price_attrs:
        if attr in ticker.info and ticker.info[attr] is not None:
            return ticker.info[attr]
    
    # Se os atributos anteriores não funcionarem, tenta extrair o preço do objeto fast_info
    fast_info = ticker.fast_info
    if hasattr(fast_info, 'last_price') and fast_info.last_price is not None:
        return fast_info.last_price

    # Se nenhum preço for encontrado, lança uma exceção
    raise Exception("Could not find valid price data")

# Dicionário que define a ferramenta get_stock_price para integração 
# com um modelo de linguagem
get_stock_price_tool = {
    'type': 'function',
    'function': {
        'name': 'get_stock_price',  # Nome da função a ser chamada
        'description': 'Get the current stock price for any symbol',  # Descrição da função
        'parameters': {  # Especificação dos parâmetros aceitos
            'type': 'object',
            'required': ['symbol'],
            'properties': {
                'symbol': {'type': 'string', 'description': 'The stock symbol (e.g., AAPL, GOOGL)'},
            },
        },
    },
}

# Prompt de exemplo que solicita o preço de uma ação específica
prompt = 'What is the current stock price of Apple?'
print('Prompt:', prompt)

# Dicionário que mapeia nomes de funções disponíveis para 
# suas implementações correspondentes
available_functions: Dict[str, Callable] = {
    'get_stock_price': get_stock_price,
}

# Chamada ao modelo de linguagem (Llama 3.1) para processar o prompt
response = ollama.chat(
    'llama3.1',
    messages=[{'role': 'user', 'content': prompt}],  # Mensagem de entrada do usuário
    tools=[get_stock_price_tool],  # Fornece ao modelo a ferramenta disponível
)

# Verifica se o modelo solicitou chamadas de função
if response.message.tool_calls:
    for tool in response.message.tool_calls:
        # Obtém a função correspondente ao nome solicitado
        if function_to_call := available_functions.get(tool.function.name):
            print('Calling function:', tool.function.name)
            print('Arguments:', tool.function.arguments)
            # Chama a função com os argumentos fornecidos pelo modelo
            print('Function output:', function_to_call(**tool.function.arguments))
        else:
            print('Function', tool.function.name, 'not found')  # Caso a função não seja encontrada
