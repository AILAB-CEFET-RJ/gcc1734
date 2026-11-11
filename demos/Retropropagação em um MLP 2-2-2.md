## Componente da demo

- **Rede:** MLP com duas entradas, dois neurônios na camada oculta e duas
  saídas, isto é, uma arquitetura 2-2-2.
- **Entrada fixa:** $\mathbf{x}=(1,2)^\top$.
- **Parâmetros iniciais:**
  ```text
  W⁽¹⁾ = [[ 1, -1 ], [ 1,  1 ]]
  b⁽¹⁾ = [0, 0]
  W⁽²⁾ = [[ 1,  1 ], [ 0,  0 ]]
  b⁽²⁾ = [0, 0]
  ```
- **Camada oculta:** calcula $\mathbf{u}=W^{(1)}\mathbf{x}+\mathbf{b}^{(1)}$
  e aplica ReLU, obtendo $\mathbf{h}=\max(0,\mathbf{u})$ componente a
  componente.
- **Saída:** calcula os logits
  $\mathbf{z}=W^{(2)}\mathbf{h}+\mathbf{b}^{(2)}$ e aplica softmax para obter
  as probabilidades das duas classes.
- **Perda:** usa entropia cruzada para o alvo escolhido. O painel permite
  alternar entre **alvo: classe 1** e **alvo: classe 2**.
- **Retropropagação:** calcula primeiro
  $\partial\ell/\partial\mathbf{z}=\mathbf{p}-\mathbf{y}$ e depois propaga
  os gradientes pela camada de saída, pela ReLU e pela camada de entrada.
- **Atualização:** modifica pesos e interceptos com
  $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla\ell$.
  A taxa $\alpha$ começa em $0{,}1$ e pode ser alterada.
- **Histórico:** o painel “Perda por ciclo” registra a perda após cada ciclo de
  atualização.

## Roteiro de uso

1. Comece com o alvo na classe 1 e avance pelas etapas com **Próximo**. A
   sequência é: combinação da camada oculta, ReLU, logits, softmax, perda,
   gradiente da saída, gradientes dos pesos e atualização.
2. Observe que, para a entrada fixa, a primeira pré-ativação oculta é negativa.
   A ReLU a transforma em zero; por isso, esse neurônio não recebe gradiente
   através da ativação nessa passagem.
3. Compare a perda quando o alvo é a classe 1 com a perda quando o alvo é a
   classe 2. A mesma distribuição de probabilidades pode produzir perdas muito
   diferentes conforme a classe observada.
4. Ao chegar à etapa de retropropagação, acompanhe a relação entre
   $\partial\ell/\partial\mathbf{z}$, os gradientes de $W^{(2)}$, os gradientes
   que chegam a $\mathbf{h}$ e os gradientes de $W^{(1)}$.
5. Altere a taxa de aprendizado antes da atualização e compare o tamanho da
   mudança nos parâmetros e o valor da perda no ciclo seguinte.
6. Use **Novo ciclo** para repetir o cálculo com os parâmetros atualizados. Use
   **Anterior**, **Próximo**, **Reproduzir** e **Reiniciar tudo** para controlar
   a animação.

## Limitações

- A entrada é um único vetor fixo, $\mathbf{x}=(1,2)^\top$. A demo não treina
  com um conjunto de exemplos nem calcula uma perda média de mini-lote.
- A arquitetura é fixa em 2-2-2. Não é possível adicionar camadas, neurônios,
  características ou classes pela interface.
- Os dois alvos possíveis são escolhidos manualmente. A demo não representa um
  processo de classificação com vários exemplos rotulados.
- A retropropagação é calculada para uma única configuração de parâmetros e um
  único exemplo por vez. Vários ciclos ilustram atualização iterativa, mas não
  constituem um treinamento completo de uma rede.
- Na implementação, a derivada da ReLU é considerada zero quando a
  pré-ativação é menor ou igual a zero. O comportamento no ponto zero é uma
  convenção computacional.
- A redução da perda desse exemplo não garante melhora em outros exemplos ou
  generalização para dados novos. A demo também não aborda validação, teste,
  regularização ou parada antecipada.
