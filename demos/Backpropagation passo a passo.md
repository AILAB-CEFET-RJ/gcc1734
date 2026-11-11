## Componente da demo

- **Grafo computacional:** a demonstração usa o grafo escalar
  ```text
  e = a × b
  d = e + c
  L = d × f
  ```
  A perda final é $L$.
- **Parâmetros iniciais:** $a=2$, $b=-3$, $c=10$ e $f=-2$. Com esses valores,
  $e=-6$, $d=4$ e $L=-8$.
- **Passagem para a frente:** as etapas calculam, nesta ordem, as entradas,
  $e$, $d$ e $L$.
- **Pergunta intermediária:** a demo perturba ligeiramente $a$ para estimar
  numericamente a derivada de $L$ em relação a esse parâmetro. Esse passo serve
  para motivar a ideia de gradiente antes da retropropagação.
- **Retropropagação:** o gradiente parte de $\partial L/\partial L=1$ e percorre
  o grafo no sentido inverso. Nas multiplicações, cada entrada recebe o valor
  da outra entrada multiplicado pelo gradiente que chega; na soma, o gradiente é
  repassado para as duas entradas.
- **Atualização:** cada parâmetro é atualizado pela regra
  ```text
  novo = antigo − α × gradiente
  ```
  A taxa $\alpha$ pode ser alterada no painel lateral.
- **Histórico:** a perda de cada ciclo é registrada no painel “Perda por ciclo”.

## Roteiro de uso

1. Comece no passo inicial e avance com **Próximo**. Observe que a passagem
   para a frente calcula a saída do grafo da esquerda para a direita.
2. No passo **A PERGUNTA**, compare a perda antes e depois da pequena alteração
   em $a$. Use esse resultado para discutir a interpretação de uma derivada como
   uma variação local da perda.
3. Continue pela etapa **BACKWARD PASS**. Observe os gradientes sendo calculados
   a partir de $L$ e fluindo na direção contrária às operações do grafo.
4. Compare as regras exibidas para a soma e para a multiplicação. Na soma, o
   gradiente é copiado; na multiplicação, ele é multiplicado pela outra entrada.
5. Na etapa **ONE STEP OF LEARNING**, altere a taxa $\alpha$ e observe como o
   tamanho da atualização muda. A atualização deve ocorrer na direção oposta ao
   gradiente.
6. Use **Novo ciclo** para repetir o procedimento com os parâmetros atualizados.
   Compare a perda registrada entre os ciclos.
7. Use **Reproduzir** para acompanhar automaticamente as etapas. **Anterior**,
   **Próximo** e **Reiniciar tudo** permitem controlar novamente a execução.

## Limitações

- O exemplo usa um grafo escalar artificial, e não uma rede neural completa.
  Ele mostra a mecânica da regra da cadeia sem introduzir matrizes, ativações ou
  um conjunto de dados.
- A perturbação de $a$ é uma estimativa numérica usada apenas para motivar a
  derivada. A retropropagação apresentada depois calcula os gradientes pela
  estrutura do grafo.
- A função $L=d\times f$ não representa uma função de perda realista e não tem
  um mínimo adequado para um problema de aprendizado. A própria interface avisa
  que a perda continua caindo por construção do exemplo.
- O ciclo usa um único conjunto de valores e uma atualização por vez. Não há
  mini-lotes, dados de treino e validação, regularização ou avaliação de
  generalização.
- A demo não aprende uma função a partir de exemplos; ela ilustra apenas o
  cálculo de gradientes e a regra de atualização de parâmetros.
