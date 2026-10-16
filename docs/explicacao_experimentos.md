# Entendendo os experimentos de Monte Carlo em Aprendizado por Reforço (Blackjack)

> Documento de apoio para quem tem pouca experiência com Aprendizado por Reforço (RL).
> Explica, do zero, o que o notebook `notebooks/experimentos.ipynb` faz, por quê, e o que os
> resultados em `resultados/` significam. Os números abaixo são os valores realmente gerados
> pela execução do notebook (`notebooks/experimentos_executado.ipynb`).

---

## 0. Glossário rápido (para quem nunca viu RL)

| Termo | O que significa aqui |
|---|---|
| **Episódio** | Uma partida completa de Blackjack, do início ("comprar cartas") até o fim (resultado: ganhou/empatou/perdeu). |
| **Estado (s)** | Uma "foto" da situação do jogador no momento de decidir: `(soma das suas cartas, carta visível do dealer, se tem ás "flexível")`. Ex.: `(16, 10, False)` = jogador tem 16 pontos, o dealer mostra 10, e o jogador não tem ás contando como 11. |
| **Ação (a)** | O que o jogador decide fazer: `STICK` (parar, não comprar mais cartas) ou `HIT` (comprar mais uma carta). |
| **Recompensa (r)** | O ambiente só paga no **fim** da partida: `+1` (vitória), `0` (empate) ou `-1` (derrota). Durante o jogo a recompensa é `0`. |
| **Retorno (G)** | A soma das recompensas de um episódio, descontada por `γ` (veja abaixo). Como só existe recompensa no final, aqui `G` é basicamente o resultado da partida. |
| **Política (π)** | A "regra de decisão" do jogador: para cada estado, diz qual ação tomar. Pode ser fixa (ex.: "sempre parar com 20+") ou aprendida. |
| **V(s)** | O valor esperado de retorno começando no estado `s` e seguindo uma política `π`. Em bom português: "se eu estiver nessa situação e jogar do meu jeito, quanto espero ganhar, em média?" |
| **Q(s,a)** | Igual ao `V(s)`, mas fixando também a primeira ação `a`. "Se eu estiver nessa situação e **primeiro fizer esta ação específica**, quanto espero ganhar depois, seguindo minha política?" |
| **γ (gama, fator de desconto)** | Um número entre 0 e 1 que indica o quanto recompensas futuras valem menos que recompensas imediatas. Em jogos de episódio curto como Blackjack, normalmente se usa `γ = 1` (sem desconto). |
| **ε (epsilon, "exploração")** | Em métodos ε-greedy, é a probabilidade de o agente tomar uma ação **aleatória** (para explorar) em vez da ação que ele acha melhor (aproveitar o que já sabe). |
| **Monte Carlo (MC)** | Família de métodos que estima `V` ou `Q` **jogando muitas partidas completas** e tirando a média dos resultados observados — sem precisar conhecer as regras/probabilidades do jogo de antemão. |

---

## 1. Contexto do trabalho

Este é o trabalho AC2 da disciplina de Aprendizado por Reforço (Ibmec), sobre **métodos de Monte Carlo**
(Sutton & Barto, 2018, capítulo 5), aplicados ao ambiente **Blackjack-v1** da biblioteca Gymnasium.

O ponto de partida foi um notebook do professor (`referencia/Aula_13_14_Monte_Carlo_RL_Didatico.ipynb`).
O grupo reaproveitou as funções desse notebook, **corrigiu alguns problemas** encontrados nele (explicados
na seção 3), e rodou uma bateria de experimentos comparando três algoritmos e vários parâmetros.

O notebook principal é `notebooks/experimentos.ipynb`. Rodá-lo gera automaticamente todas as tabelas
(`.csv`) e gráficos (`.png`) da pasta `resultados/`, prontos para os slides da apresentação.

---

## 2. O ambiente: Blackjack-v1

Regras resumidas do jogo (versão simplificada usada pelo ambiente, sem "dobrar" ou "dividir"):

- O jogador vê sua própria soma de cartas e **uma única carta** do dealer (a outra fica escondida).
- A cada turno, o jogador escolhe `HIT` (compra carta) ou `STICK` (para).
- Se a soma do jogador passar de 21, ele "estoura" e perde na hora.
- Quando o jogador para, o dealer compra cartas automaticamente enquanto sua soma for menor que 17.
- No final, compara-se quem ficou mais perto de 21 sem estourar.
- Ás pode valer 1 ou 11 (quando vale 11 sem estourar, chama-se "ás usável/flexível").

**Importante:** o estado considera três coisas: `(soma do jogador, carta do dealer, ás usável?)`. Como a
soma do jogador pode ir de 4 a 21, e a carta do dealer de 1 (ás) a 10, existem **280 estados possíveis**
(18 somas × 10 cartas do dealer, sem ás usável, mais 10 somas × 10 cartas, com ás usável). Isso é
relevante para o Experimento E (seção 7.6) — um bug no notebook original ignorava boa parte desses
estados.

A recompensa só existe no fim do episódio (`+1`/`0`/`-1`), e como se usa `γ = 1`, **todo estado visitado
numa partida recebe o mesmo retorno**: o resultado final daquela partida.

---

## 3. Ajustes feitos em relação ao notebook do professor (e por quê)

O grupo leu o código original com cuidado e encontrou pontos que distorciam a comparação entre os
algoritmos. Todos foram corrigidos e estão marcados no notebook com `# [ALTERADO]` ou `[NOVO]`:

1. **Estados com soma < 12 (o problema mais importante).** O ambiente tem 280 estados possíveis (somas de
   4 a 21), mas o algoritmo "Exploring Starts" (ES) do notebook original só sorteava estados com soma
   entre 12 e 21, e avaliava a política usando `STICK` (parar) para qualquer estado "não aprendido" — ou
   seja, para somas abaixo de 12. Isso é um problema sério: parar o jogo com 8, 9 ou 10 pontos é uma
   decisão péssima (na prática, você está desistindo de uma mão em que ainda pode melhorar sem risco
   nenhum de estourar). Isso **prejudicava artificialmente o ES** na comparação com o outro algoritmo, que
   aprendia esses estados. A correção: o ES passou a sortear todos os 280 estados, e qualquer estado "não
   aprendido" (por qualquer dos algoritmos) usa `HIT` como padrão quando a soma é menor que 12 (nunca
   estoura nesse caso). O efeito numérico dessa correção está demonstrado nas seções 7.2 e 7.6 abaixo — é
   grande.
2. **Reprodutibilidade.** O código original não fixava sementes aleatórias de forma consistente. Aqui,
   todo experimento usa sementes fixas e roda com **3 sementes diferentes** (ou 2 no modo rápido), reportando
   média ± erro padrão — porque uma única rodada de Monte Carlo tem bastante variância (sorte do baralho).
3. **"Primeira visita" calculada errado.** O algoritmo *First-Visit* deve considerar apenas a primeira vez
   que um estado aparece dentro do episódio. No código original, o teste rodava "de trás para frente" e
   acabava marcando a *última* ocorrência, não a primeira. Isso foi corrigido. Na prática (seção 7.1),
   **nenhum estado se repete dentro de uma partida de Blackjack**, então, neste jogo específico,
   First-Visit e Every-Visit dão exatamente o mesmo resultado — o bug não mudava os números finais, mas
   estava conceitualmente errado.
4. **Métrica de desempenho.** O notebook original media desempenho pela média (não ponderada) dos valores
   `V(s)` estimados. Aqui, o desempenho de cada política é medido de forma mais direta e realista: **o
   retorno médio por partida jogada** (± erro padrão), além das porcentagens de vitória/empate/derrota —
   que é o que realmente importa para avaliar "essa política é boa?".
5. **ε-greedy mais flexível.** A implementação aceita tanto um `ε` fixo quanto uma função de decaimento
   (reduzir `ε` ao longo do treino), com gerador aleatório de sementes controladas.
6. **Desempenho do código.** Pequenas otimizações (evitar listas desnecessárias, deixar a checagem de
   primeira visita mais rápida) — não muda os resultados, só a velocidade de execução.

---

## 4. Os três algoritmos estudados

Todos pertencem à família *Monte Carlo*: aprendem jogando muitas partidas completas e tirando médias dos
retornos observados — sem nunca precisar saber as probabilidades internas do jogo.

### 4.1 First-Visit MC Prediction ("eu já tenho uma política, quão boa ela é?")

Dado uma política **fixa** (por exemplo, "pare só com 20 ou mais"), esse método estima `V(s)` para cada
estado: joga muitas partidas, e para cada estado, faz a média dos retornos observados **a partir da
primeira vez que aquele estado apareceu** em cada partida. É "prediction" porque não muda a política —
só avalia o quão boa ela é.

### 4.2 Monte Carlo com Exploring Starts (ES) ("eu quero descobrir a melhor política, controlando o início")

Aqui o objetivo é **melhorar** a política, não só avaliá-la. O problema é que, para aprender `Q(s,a)`
(quanto vale fazer a ação `a` no estado `s`), é preciso experimentar **todas** as combinações de estado e
ação — inclusive aquelas que uma política gulosa normalmente nunca escolheria. A solução do ES é "trapacear"
só no início: cada partida começa forçada num par `(estado, ação)` sorteado aleatoriamente (um
"início exploratório"). Depois do primeiro passo, o jogo segue a política aprendida até ali. A cada
atualização de `Q`, a política vira imediatamente gulosa (escolhe sempre a ação de maior `Q` conhecido).
Isso é o princípio de **Generalized Policy Iteration** (GPI): avaliar e melhorar a política, alternadamente,
episódio a episódio.

Como o Gymnasium não deixa escolher livremente o estado inicial de uma partida, o notebook usa um
"wrapper" (`BlackjackES`) que força as cartas certas na mão do jogador e do dealer para simular o início
exato desejado.

**Limitação prática importante do ES:** ele só funciona porque estamos num simulador, onde é possível
"forçar" o estado inicial de um episódio. Numa situação real (um robô, por exemplo), isso geralmente é
impossível — não dá para forçar o mundo real a começar num estado arbitrário. Por isso ES é mais um
recurso teórico/de simulação do que algo aplicável fora dela.

### 4.3 On-Policy First-Visit MC Control com ε-greedy ("eu quero melhorar a política, mas sem poder escolher o início")

Alternativa ao ES para o mesmo objetivo (melhorar a política), sem depender de poder forçar o estado
inicial. A ideia: em vez de ser 100% gulosa, a política usada para jogar é **ε-soft** — com probabilidade
`1-ε` ela faz a ação que parece melhor até agora ("aproveitamento"/*exploitation*), e com probabilidade `ε`
faz uma ação aleatória ("exploração"/*exploration*). Isso garante que, cedo ou tarde, toda ação em todo
estado acabe sendo experimentada, mesmo sem forçar o início.

Diferente do ES, aqui a melhoria da política é **implícita**: não existe um passo separado de "agora eu
atualizo a política" — o próximo episódio já usa automaticamente o `Q` mais recente dentro da regra
ε-greedy. Um detalhe teórico importante: como a política continua sendo ε-soft (nunca 100% gulosa), esse
método converge para a **melhor política ε-soft possível**, não para a política ótima "pura" (`π*`). Por
isso, para comparar com as outras políticas, o notebook extrai e avalia a política **gulosa** derivada do
`Q` aprendido (ou seja, "esquecendo" a parte aleatória de exploração na hora de medir o desempenho final).

---

## 5. As políticas comparadas (First-Visit MC Prediction)

Antes de aprender políticas do zero (ES e On-Policy), o grupo avaliou quatro políticas **fixas**, para ter
pontos de referência:

| Política | Regra |
|---|---|
| **Simples ≥20 (do professor)** | Só para (`STICK`) com soma 20 ou 21; compra carta (`HIT`) em qualquer outro caso. Baseline clássico do livro-texto, propositalmente ruim. |
| **Conservadora ≥17** | Imita a regra do próprio dealer: compra carta enquanto a soma for menor que 17. |
| **Estratégia básica** | Aproximação da estratégia básica real de Blackjack (sem dobrar/dividir): leva em conta se a mão tem ás flexível e se a carta do dealer é "fraca" (2–6) ou "forte" (7–10/A). |
| **Aleatória 50/50** | Decide `HIT` ou `STICK` por sorteio, 50% de chance cada. Baseline "burro" de comparação. |

Um detalhe de implementação a notar: no ambiente, o **ás do dealer aparece codificado como o número 1**.
Um teste ingênuo tipo `dealer >= 7` trataria o ás como "carta fraca", o que é errado (ás é das cartas mais
fortes para o dealer). Por isso o código trata explicitamente a faixa `2 <= dealer <= 6` como "dealer
fraco".

---

## 6. Resultados — First-Visit MC Prediction (seção 3 do notebook)

### 6.1 Verificação: First-Visit é igual a Every-Visit neste jogo?

O grupo rodou 50.000 partidas com duas políticas diferentes e contou quantas tiveram algum estado
repetido dentro do mesmo episódio:

| Política | Episódios com estado repetido (de 50.000) |
|---|---|
| Aleatória 50/50 | 0 |
| Conservadora ≥17 | 0 |

**Conclusão em linguagem simples:** numa partida de Blackjack, como a soma do jogador só pode aumentar (ou
a partida acaba), o mesmo estado nunca se repete dentro de uma partida. Por isso, "olhar só a primeira
visita" (First-Visit) e "olhar todas as visitas" (Every-Visit) dão exatamente o mesmo resultado aqui — a
distinção teórica entre os dois métodos simplesmente não faz diferença neste jogo específico.

### 6.2 Demonstração do viés causado por ignorar somas < 12

Essa tabela mostra o efeito de **avaliar a política de duas formas diferentes**: (a) do jeito do notebook
original — só guardando decisões para somas 12–21 e usando `STICK` para qualquer soma menor, e (b) do
jeito corrigido — com a política completa, decidindo `HIT` normalmente abaixo de 12.

| Política | Forma de avaliação | Retorno médio | Vitórias % | Empates % | Derrotas % |
|---|---|---|---|---|---|
| Simples ≥20 | original (STICK abaixo de 12) | −0,362 | 29,7% | 4,4% | 65,9% |
| Simples ≥20 | corrigida (HIT abaixo de 12) | −0,350 | 29,7% | 5,5% | 64,8% |
| Estratégia básica | original (STICK abaixo de 12) | **−0,134** | 39,9% | 6,9% | 53,2% |
| Estratégia básica | corrigida (HIT abaixo de 12) | **−0,047** | 43,1% | 9,1% | 47,8% |

**Leitura em linguagem simples:** para a política "Simples ≥20" (que já é ruim, pois ela compra carta até
chegar em 19 de qualquer jeito), o bug quase não muda nada — a política raramente chega a parar "sem
querer" em somas baixas. Mas para uma política boa ("Estratégia básica"), o bug é muito prejudicial: o
retorno médio quase **triplica de ruim** (de −0,047 para −0,134) só porque, em estados de soma baixa, a
versão "errada" manda parar o jogo (desistir de melhorar a mão) em vez de comprar carta — um erro grosseiro
que qualquer jogador humano evitaria. Isso confirma que o bug original não é um detalhe menor: ele distorce
bastante a avaliação de políticas boas.

### 6.3 Desempenho das quatro políticas (200.000 episódios cada)

| Política | Retorno médio | Erro padrão | Vitórias % | Empates % | Derrotas % |
|---|---|---|---|---|---|
| **Estratégia básica** | **−0,0406** | 0,0021 | 43,46% | 9,02% | 47,52% |
| Conservadora ≥17 | −0,0728 | 0,0021 | 41,26% | 10,20% | 48,54% |
| Simples ≥20 (professor) | −0,3519 | 0,0020 | 29,64% | 5,54% | 64,82% |
| Aleatória 50/50 | −0,3962 | 0,0020 | 28,12% | 4,15% | 67,74% |

**Leitura em linguagem simples:** em Blackjack, até a melhor estratégia possível perde dinheiro em média a
longo prazo (a "vantagem da casa" é inerente ao jogo) — por isso todos os retornos médios são negativos.
O que importa é a comparação **relativa**: a "Estratégia básica" é claramente a melhor das quatro (perde
menos, em média, por partida), seguida pela "Conservadora ≥17". A política "Simples ≥20" do professor e a
"Aleatória 50/50" são bem piores — intuitivamente, isso faz sentido: parar de comprar cartas só com 20+
(deixando de comprar quando se tem, digamos, 14) é uma estratégia ruim, e jogar por sorteio (50/50) é
ainda pior.

Os heatmaps gerados (`resultados/fv_heatmap_*.png`) mostram visualmente `V(s)` para cada política, separando
os casos "com ás usável" e "sem ás usável", cruzando a soma do jogador (eixo vertical) com a carta visível
do dealer (eixo horizontal). O gráfico `resultados/fv_convergencia_estados_chave.png` mostra como a
estimativa de `V(s)` em alguns estados específicos (ex.: soma 16 contra dealer com 10) vai se estabilizando
à medida que mais episódios são simulados — um exemplo visual de "Lei dos Grandes Números" em ação: quanto
mais partidas, mais a média amostral se aproxima do valor real esperado.

---

## 7. Experimentos de controle (seção 6 do notebook): ES × On-Policy ε-greedy

Protocolo usado em todos os experimentos desta seção: cada configuração é treinada com **3 sementes
diferentes** (para medir variação por acaso), avaliada em pontos de checagem (*checkpoints*, de 5 mil até
200 mil episódios de treino) e, ao final do treino completo, avaliada com 50.000 partidas de teste. Os
números reportados são médias entre as 3 sementes (± erro padrão/desvio, quando aplicável).

### 7.1 Experimento B — ES vs On-Policy, variando ε

Pergunta: **ES (que precisa de início forçado) aprende uma política tão boa quanto o On-Policy ε-greedy
(que não precisa)? E qual valor de ε funciona melhor?**

| Configuração | Retorno médio final | Desvio entre sementes | Vitórias % | Empates % | Derrotas % |
|---|---|---|---|---|---|
| ES | −0,0411 | 0,0071 | 43,49% | 8,90% | 47,61% |
| On-Policy ε=0,01 | −0,0720 | 0,0013 | 42,33% | 8,15% | 49,53% |
| On-Policy ε=0,05 | −0,0496 | 0,0036 | 43,23% | 8,57% | 48,19% |
| **On-Policy ε=0,1** | **−0,0453** | 0,0018 | 43,41% | 8,65% | 47,94% |
| On-Policy ε=0,2 | −0,0477 | 0,0011 | 43,42% | 8,39% | 48,19% |
| On-Policy ε=0,4 | −0,0510 | 0,0072 | 43,43% | 8,04% | 48,53% |

**Leitura em linguagem simples:**
- O **ES** termina com o melhor resultado entre todas as configurações (−0,0411), mas também é o que mais
  varia de uma semente para outra (desvio 0,0071) — ele depende mais de "sorte" na hora de explorar.
- Dentro do On-Policy, **ε muito pequeno (0,01) é o pior resultado** (−0,0720): o agente explora pouco, então
  demora muito para experimentar ações que, por acaso, parecem ruins no começo (mas não são) — ele fica
  "preso" perto do que já conhece.
- **ε=0,1 é o melhor ponto do On-Policy** aqui: explora o suficiente para descobrir boas ações, sem
  desperdiçar jogadas demais em ações aleatórias ruins.
- **ε grande demais (0,4) piora de novo**: o agente passa 40% do tempo jogando aleatoriamente mesmo depois
  de já ter aprendido o suficiente, o que atrapalha o desempenho final.
- Ou seja, existe um "ponto ideal" de exploração: pouco demais e muito demais são ambos ruins — um
  padrão clássico em RL, conhecido como o dilema **exploração × aproveitamento**.

O gráfico `resultados/exp_efeito_epsilon.png` mostra exatamente essa curva em formato de "U" (ou quase):
o desempenho piora nas pontas (ε muito baixo ou muito alto) e melhora no meio, com uma linha horizontal
marcando o desempenho do ES e da "Estratégia básica" para comparação.

O gráfico `resultados/exp_curvas_aprendizado.png` mostra como cada configuração evolui ao longo do treino
(eixo X em escala logarítmica, de 5 mil a 200 mil episódios): dá para ver visualmente qual método aprende
mais rápido e qual estabiliza em qual patamar.

### 7.2 Experimento C — decaimento de ε (começar explorando mais, depois menos)

Ideia testada: começar com `ε=0,5` (bastante exploração no início, quando ainda não se sabe nada) e ir
reduzindo linearmente até `ε=0,01` (quase só aproveitamento, perto do fim do treino) — uma estratégia comum
em RL para "ter o melhor dos dois mundos".

| Configuração | Retorno médio final | Desvio entre sementes |
|---|---|---|
| On-Policy ε linear 0,5→0,01 | −0,0503 | 0,0005 |
| (comparar com) On-Policy ε=0,1 fixo | −0,0453 | 0,0018 |

**Leitura em linguagem simples:** neste experimento específico, o decaimento de ε **não superou** o ε fixo
em 0,1 — ficou ligeiramente pior em retorno médio, embora bem mais estável entre sementes (desvio de
apenas 0,0005, o mais baixo de todos os experimentos). Isso sugere que, para este jogo e esse cronograma de
decaimento, um `ε=0,1` fixo já é "bom o suficiente" durante todo o treino — o Blackjack é um jogo
relativamente simples (poucos estados, episódios curtos), então não há tanta vantagem em ajustar
dinamicamente a exploração como haveria em ambientes mais complexos.

### 7.3 Experimento D — efeito do fator de desconto γ

Ideia testada: em vez do padrão `γ=1` (recompensas futuras valem o mesmo que recompensas imediatas),
testar `γ=0,9` (recompensas mais distantes no tempo valem um pouco menos).

| Configuração | Retorno médio final | Desvio entre sementes |
|---|---|---|
| ES (γ=1) | −0,0411 | 0,0071 |
| ES γ=0,9 | −0,0429 | 0,0037 |
| On-Policy ε=0,1 (γ=1) | −0,0453 | 0,0018 |
| On-Policy ε=0,1 γ=0,9 | −0,0446 | 0,0068 |

**Leitura em linguagem simples:** a diferença entre `γ=1` e `γ=0,9` é pequena nos dois algoritmos — o que
faz sentido, porque no Blackjack os episódios são curtos (poucas jogadas) e a recompensa só existe no
final. Com `γ=0,9`, cada passo adicional (cada `HIT`) "desconta" um pouco o valor do resultado final,
o que tende a incentivar levemente a parar mais cedo (menos HITs) — mas como as partidas já são curtas,
o efeito prático é pequeno. Por isso `γ=1` é considerado o "natural" para este jogo: não há razão teórica
para descontar o futuro quando os episódios já são curtos e bem definidos.

### 7.4 Experimento E — o quanto o bug original (seção 3, item 1) realmente prejudicava o ES

Esse experimento isola e mede exatamente o impacto do bug descrito na seção 3 (item 1), comparando três
variantes do mesmo algoritmo ES:

| Variante | Retorno médio | Desvio |
|---|---|---|
| ES original (só sorteia somas 12–21; usa STICK nos demais estados) | **−0,1334** | 0,0024 |
| ES sorteia 12–21, mas usa HIT (em vez de STICK) abaixo de 12 | −0,0402 | 0,0038 |
| **ES corrigido (sorteia todos os 280 estados, 4–21)** | **−0,0411** | 0,0071 |

**Leitura em linguagem simples:** essa é a confirmação mais direta do problema apontado na seção 3. O ES
"do jeito que estava no notebook original" tem um desempenho **mais de 3x pior** (−0,1334) do que o ES
corrigido (−0,0411) — e a causa não é o algoritmo em si, é simplesmente a política "de reserva" usada para
estados não aprendidos: mandar `STICK` (parar) quando a soma é baixa é uma decisão horrível, que nenhuma
pessoa tomaria jogando Blackjack de verdade. Curiosamente, a segunda variante (sortear só 12–21, mas usar
`HIT` como padrão abaixo de 12) já chega quase no mesmo patamar do ES corrigido — ou seja, **a maior parte
do problema vinha da política padrão usada para estados não aprendidos**, não do fato de não sortear
estados abaixo de 12 durante o treino. Isso é uma boa lição de depuração: às vezes o "bug" não está no
algoritmo principal, mas num detalhe pequeno (um valor padrão/`default`) que passa despercebido.

### 7.5 Divergência entre as políticas aprendidas

Depois de treinar ES e On-Policy (ε=0,1), o grupo comparou, estado por estado (somente os 200 estados com
soma entre 12 e 21), em quantos deles cada par de políticas discorda sobre a ação a tomar:

| Comparação | % dos estados em que discordam |
|---|---|
| ES × On-Policy | 13,5% |
| ES × Estratégia básica | 3,5% |
| On-Policy × Estratégia básica | 11,0% |

**Leitura em linguagem simples:** o ES aprendeu uma política mais parecida com a "Estratégia básica" de
referência (só 3,5% de discordância) do que o On-Policy (11,0% de discordância). Isso é coerente com o
Experimento B, onde o ES teve o melhor retorno médio: uma política mais parecida com a estratégia básica
correta tende, de fato, a jogar melhor. As discordâncias que sobram geralmente aparecem em **estados raros**
(por exemplo, somas muito específicas contra certas cartas do dealer), onde há pouquíssimas partidas de
treino para "calibrar" bem a decisão — isso é chamado de **ruído amostral**: com poucas observações, a
média calculada pode não refletir ainda o valor real.

Os arquivos `resultados/politica_es.png` e `resultados/politica_onpolicy_eps01.png` mostram visualmente
(verde = HIT, vermelho = STICK) a política aprendida por cada algoritmo, lado a lado (com e sem ás
usável). Os arquivos `resultados/v_es.png` e `resultados/v_onpolicy_eps01.png` mostram os valores
`V(s) ≈ max_a Q(s,a)` aprendidos por cada um.

---

## 8. Resumo das principais conclusões (em linguagem simples)

1. **O ambiente tem 280 estados**, não só os 200 (somas 12–21) que o notebook original considerava — isso
   importa principalmente para o algoritmo ES, cujo desempenho **triplicou** (de −0,1334 para −0,0411) só
   com a correção.
2. **First-Visit = Every-Visit neste jogo específico**, porque nenhum estado se repete dentro de uma
   partida de Blackjack — a distinção teórica existe, mas não muda nenhum número aqui.
3. Entre as políticas fixas, a **Estratégia básica** é a melhor (−0,041 de retorno médio), bem à frente da
   política "ingênua" do professor (−0,352) e da aleatória (−0,396).
4. **ES e On-Policy ε-greedy chegam a desempenhos parecidos** (ambos em torno de −0,04 a −0,05) depois de
   corrigidos — mas usam estratégias bem diferentes para chegar lá: ES "trapaceia" controlando o início de
   cada episódio (só possível em simulação), enquanto On-Policy explora de forma mais realista com ε-greedy
   (aplicável também fora de simuladores).
5. Existe um **ε ideal (aproximadamente 0,1)** para o On-Policy: muito pouco (0,01) ou muito (0,4) de
   exploração pioram o resultado final — o clássico dilema exploração × aproveitamento.
6. O **decaimento de ε** testado não superou o ε fixo em 0,1 neste caso (ambiente simples, poucos
   estados) — mas tornou o aprendizado mais consistente entre diferentes sementes.
7. O **fator de desconto γ** teve efeito pequeno (1,0 vs 0,9), como esperado em episódios curtos como o
   Blackjack, onde γ=1 é a escolha natural.
8. As políticas aprendidas (ES, On-Policy) e a Estratégia básica divergem principalmente em **estados
   raros**, onde há pouco treino para calibrar bem a decisão.

---

## 9. Onde encontrar cada arquivo gerado

Todos os arquivos abaixo são criados automaticamente ao rodar `notebooks/experimentos.ipynb` (ficam em
`resultados/`):

| Arquivo | Conteúdo |
|---|---|
| `demo_vies_somas_baixas.csv` | Tabela da seção 6.2 (viés ao avaliar com `dict` 12–21 vs função completa) |
| `fv_desempenho_politicas.csv` | Tabela da seção 6.3 (desempenho das 4 políticas fixas) |
| `fv_heatmap_*.png` | Heatmaps de `V(s)` de cada política fixa |
| `fv_convergencia_estados_chave.png` | Convergência de `V(s)` em 3 estados-chave, ao longo do treino |
| `exp_curvas_aprendizado.png` | Curvas de aprendizado do Experimento B (ES × On-Policy, vários ε) |
| `exp_efeito_epsilon.png` | Gráfico "efeito de ε" do Experimento B |
| `exp_E_vies_somas_baixas.csv` | Tabela do Experimento E (seção 7.4) |
| `resumo_experimentos.csv` | Tabela consolidada de todos os experimentos B/C/D (seção 7.1–7.3) |
| `experimentos_final_por_seed.csv` | Mesma coisa, mas sem agregar — uma linha por semente |
| `experimentos_curvas.csv` | Dados brutos das curvas de aprendizado (todas as configurações, todos os checkpoints) |
| `divergencia_politicas.csv` | Tabela da seção 7.5 |
| `politica_es.png`, `politica_onpolicy_eps01.png` | Heatmaps das políticas aprendidas por ES e On-Policy |
| `v_es.png`, `v_onpolicy_eps01.png` | Heatmaps de `V(s)` aprendidos por ES e On-Policy |
| `politicas_aprendidas.csv` | Ação (HIT/STICK) de cada política, estado a estado, para os 280 estados |
| `valores_V_por_estado.csv` | `V(s)` de cada política/algoritmo, estado a estado |
| `verificacoes.csv` | Pequenas checagens de sanidade do código (ex.: a seção 6.1) |

Para rodar tudo de novo: siga as instruções do `README.md` do projeto (criar o ambiente virtual Python,
instalar `requirements.txt`, abrir `notebooks/experimentos.ipynb` no Jupyter e rodar as células em ordem).
Existe um "modo rápido" (`MC_RAPIDO=1`) que faz tudo em ~1–2 minutos, com números menos precisos — útil só
para testar se o código roda sem erros, não para os resultados finais da apresentação (que usam o modo
completo, ~20–25 minutos).
