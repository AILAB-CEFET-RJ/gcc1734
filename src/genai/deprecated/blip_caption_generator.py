from transformers import pipeline
from PIL import Image
import requests
import io

def gerar_legenda_imagem_blip(caminho_ou_url_imagem: str):
    """
    Gera uma legenda (descrição) para uma imagem usando o modelo BLIP.

    Args:
        caminho_ou_url_imagem (str): O caminho para um arquivo de imagem local
                                     ou uma URL de uma imagem na internet.
    """
    print("--- Inicializando o Modelo BLIP ---")
    print("Isso pode levar alguns minutos na primeira execução para baixar o modelo.")

    try:
        # Carrega o pipeline de "image-to-text" com o modelo BLIP.
        # "Salesforce/blip-image-captioning-base" é uma boa escolha para começar.
        image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        print("Modelo BLIP carregado com sucesso!")

        # Carrega a imagem a partir de um caminho local ou URL
        if caminho_ou_url_imagem.startswith("http://") or caminho_ou_url_imagem.startswith("https://"):
            response = requests.get(caminho_ou_url_imagem)
            response.raise_for_status() # Lança um erro para status HTTP ruins
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            print(f"Imagem carregada de URL: {caminho_ou_url_imagem}")
        else:
            image = Image.open(caminho_ou_url_imagem).convert("RGB")
            print(f"Imagem carregada de arquivo local: {caminho_ou_url_imagem}")

        # Gera a legenda da imagem
        print("Gerando legenda para a imagem...")
        caption_results = image_captioner(image)

        # O resultado é uma lista de dicionários, pegamos o texto gerado
        legenda_gerada = caption_results[0]['generated_text']

        print("\n--- Resultado ---")
        print(f"A imagem foi descrita como: **'{legenda_gerada}'**")

    except FileNotFoundError:
        print(f"Erro: Imagem não encontrada no caminho '{caminho_ou_url_imagem}'.")
        print("Certifique-se de que o caminho do arquivo está correto ou que a URL é válida.")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar a imagem da URL '{caminho_ou_url_imagem}': {e}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        print("Verifique se todas as dependências estão instaladas corretamente e se a imagem é válida.")

# --- Exemplos de Uso ---
if __name__ == "__main__":
    # Exemplo 1: Usando uma URL de imagem (mais fácil para testar rapidamente)
    # Uma imagem simples de um cachorro:
    # url_imagem_exemplo = "https://www.petz.com.br/blog/wp-content/uploads/2022/09/cachorro-vomita-no-carro-3.jpg"
    # gerar_legenda_imagem_blip(url_imagem_exemplo)

    print("\n" + "="*50 + "\n")

    # Exemplo 2: Usando um caminho de arquivo local
    # IMPORTANTE: Para este exemplo funcionar, você precisa ter um arquivo de imagem
    # chamado 'minha_imagem_teste.jpg' (ou outro nome) no mesmo diretório que este script.
    # Você pode baixar uma imagem qualquer e renomeá-la.
    # Por exemplo, baixe a imagem do cachorro do link acima e salve-a como 'minha_imagem_teste.jpg'
    
    # Substitua 'minha_imagem_teste.jpg' pelo caminho real do seu arquivo local
    caminho_imagem_local = "./img/cachorros-no-carro.jpg" 
    gerar_legenda_imagem_blip(caminho_imagem_local) # Descomente para testar este exemplo