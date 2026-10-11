## Componente da demo

- **Painel principal:** a grade 4×3 da figura da nota, com o terminal +1 em (4,3), o terminal −1 em (4,2), o obstáculo em (2,2) e o agente (🤖) partindo de (1,1). As setas do teclado escolhem a ação pretendida. Os botões ↑ ← ↓ → servem para telas de toque.
- **Estocasticidade:** a ação pretendida ocorre com a probabilidade de sucesso (0,8 por padrão). Com a probabilidade restante, o agente desvia 90° para a esquerda ou para a direita, metade para cada lado. Movimentos contra a parede ou o obstáculo mantêm o agente na célula.
- **Painel "Episódio":** número de passos, retorno sem desconto e retorno descontado. O botão **Reiniciar** (ou a tecla **R**) volta ao estado inicial.
- **Painel "Ambiente (T e R)":**
  - Sliders de probabilidade de sucesso, *living reward* e γ. Mudar qualquer um deles reinicia o episódio.
  - Tabela com T(s,a,s′) do estado atual para a última ação escolhida. Mostra as três saídas possíveis, suas probabilidades, o estado de destino e a recompensa.
- **Painel "Política π":**
  - Seletor com quatro opções: ocultar a política, π* determinística, estocástica ε-greedy sobre π* e estocástica uniforme.
  - Na política determinística, cada célula mostra uma seta grande. Nas estocásticas, mostra as setas com π(a|s) em porcentagem.
  - Slider de ε, usado só na política ε-greedy.
  - Seletor para exibir V*(s) em cada célula.
  - O botão **Agente segue π** (ou a tecla **Espaço**) dá um passo seguindo a política selecionada. O botão **Auto** repete esse passo continuamente.
- **Registro:** lista dos últimos passos, com o estado, a ação pretendida, o resultado e a recompensa. Os desvios estocásticos aparecem destacados.

## Roteiro de uso

1. Ponha a probabilidade de sucesso em 1,0 e jogue manualmente. O ambiente fica determinístico: a ação pretendida é sempre a executada.
2. Volte a 0,8 e tente repetir o mesmo caminho várias vezes. Os resultados mudam de uma tentativa para outra, e o registro mostra os desvios. Compare os retornos entre os episódios.
3. Mostre π* e compare com a figura correspondente da nota. Rode o modo **Auto** algumas vezes: a política é a mesma, mas as trajetórias e os retornos diferem.
4. Troque para a política ε-greedy e aumente ε. Peça ao aluno que leia π(a|s) em uma célula e preveja o que o agente fará. Depois mostre a política uniforme como caso extremo.
5. Ative V*(s) e mude o *living reward* e γ. Observe como π* e os valores se reorganizam. A nota indica, por exemplo, que a ação ótima em (4,1) muda de Oeste para Sul quando γ cai de 0,9 para 0,5.

## Limitações

- O estado inicial é fixo em (1,1) e o ambiente é sempre o Mundo Grade 4×3.
- π* é calculada na própria página por iteração de valor, a partir de T e R. O agente **não aprende** nada: não há Q-learning nem SARSA nesta demo.
- Em caso de empate entre ações, π* escolhe a primeira na ordem Norte, Leste, Sul, Oeste.
- As setas do teclado sempre executam a ação escolhida pelo usuário, independentemente da política exibida. A política só age quando se usa **Espaço** ou **Auto**.
- Com "Ocultar política" selecionado, **Espaço** e **Auto** seguem π*.
