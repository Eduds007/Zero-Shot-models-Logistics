# Aprendizado contínuo de detecção de objetos no robô FML Tugger

Dissertação de mestrado (TUM, FML). O robô captura imagens, o SAM3 encontra
uma classe nova, um humano revisa os casos incertos e um único YOLO é
retreinado para aprender a classe sem esquecer as antigas. Um gate
estatístico decide se o modelo vai para o robô, que o troca sozinho.

- Pipeline e como rodar: [`dataset/pipeline/README.md`](dataset/pipeline/README.md)
- Registro completo dos experimentos: [`dataset/pipeline/CATASTROPHIC_FORGETTING_RESULTS.md`](dataset/pipeline/CATASTROPHIC_FORGETTING_RESULTS.md)
- Resultados brutos: `dataset/pipeline/self_training/results_*.json`

Salvo indicação, os números são AP50, média de 3 seeds. As classes do LOCO
são medidas no LOCO val (2277 imagens de armazéns nunca vistos no treino).
Estado em 05/11/2026.

## Parte 1. Resultados atuais

O ciclo completo funcionou no robô: captura, SAM3, revisão humana, treino,
gate estatístico e troca do modelo ao vivo. O modelo promovido detecta 7
classes e perde 2,4% na média do LOCO. Retirar a revisão humana não mudou de
forma mensurável a detecção da classe nova, mas nenhum dos 3 modelos sem
revisão passou no gate, por causa de pallet_truck. Com revisão, passaram 3 de
6 modelos.

### Imagens capturadas

| sessão | frames | uso |
|---|---:|---|
| 16/09 | 180 | pipeline |
| 17/09 | 340 | pipeline (treino de iw_hub e do braço; 30 frames viraram teste de iw_hub) |
| 05/10 | 390 | dia reservado para teste: nunca entra no treino |
| 02/11, manhã | 172 | pipeline (gravação nova) |
| **total no pipeline** | **1082** | |
| 02/11, tarde | 391 | fora do pipeline; 88 após tirar duplicatas; sem rótulos |

### Do frame capturado ao treino: braço KUKA

| etapa | com revisão | sem revisão |
|---|---:|---:|
| frames analisados pelo SAM3 | 1082 | 1082 |
| passaram no filtro do SAM3 (confiança média > 0,5) | 221 | 221 |
| depois de tirar o dia de teste (05/10) | 162 | 162 |
| decisão humana (161 revisados + 1 aprovado automaticamente) | 116 aprovados, 46 rejeitados | nenhuma |
| **no lote, depois de tirar imagens de teste do iw_hub** | **107** | **140** (33 rejeitados pelo humano) |

Os dois lotes têm também os 83 frames de iw_hub. As caixas são as mesmas do
SAM3: a revisão só aprovou ou rejeitou frames, não corrigiu caixas.

### Pipeline com e sem revisão humana

| classe | modelo antigo | com revisão | sem revisão | seeds com diferença significativa (IC 95%) |
|---|---:|---:|---:|---|
| **braço KUKA** | não detecta | **.542** | **.555** | 0 de 3 |
| iw_hub (lab + Roboflow) | não detecta | .542 | .592 | 1 de 3, sem revisão melhor |
| SLC | .385 | .407 | .392 | 3 de 3, sem revisão pior |
| forklift | .262 | .242 | .241 | 0 de 3 |
| pallet | .632 | .644 | .643 | 1 de 3, sem revisão pior |
| stillage | .496 | .476 | .491 | 3 de 3, sem revisão melhor |
| pallet_truck | .278 | .253 | .236 | 2 de 3, sem revisão pior |
| **média LOCO** | **.411** | **.404** | **.401** | |

- No braço, as diferenças ficam dentro do ruído. A mesma configuração com
  revisão, treinada duas vezes, variou até ±.02 por seed (rodada real .562 /
  .524 / .520; repetição .578 / .545 / .503).
- **Gate** (contra o modelo antigo): sem revisão, **0 de 3** seeds passam, os
  três reprovados em pallet_truck (.233 / .242 / .234 contra .278). Com
  revisão, **3 de 6**: 2 de 3 na rodada real e 1 de 3 na repetição, com
  reprovações em stillage e pallet_truck.
- O resultado do gate muda entre repetições da mesma configuração (2 de 3
  contra 1 de 3). A diferença 3 de 6 contra 0 de 3 aponta a favor da revisão,
  mas a amostra é pequena, e a causa da queda em pallet_truck não foi isolada.
- As 84 imagens de teste do braço foram pré-selecionadas pelo SAM3 e não
  incluem imagens sem o braço. Isso pode favorecer o treino sem revisão e
  superestimar o AP nos dois casos.
- Os dois braços diferem em qualidade e em quantidade de frames (107 contra
  140).

### Rodada real do pipeline (02/11)

| classe | modelo antigo | seed 0 (promovido) | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| SLC | .385 | .408 | .403 | .410 |
| forklift | .262 | .225 | .213 | .236 |
| pallet | .632 | .644 | .641 | .644 |
| stillage | .496 | .479 | .470 (reprova) | .481 |
| pallet_truck | .278 | .251 | .261 | .259 |
| iw_hub | — | .549 | .550 | .535 |
| braço KUKA | — | .562 | .524 | .520 |
| **média LOCO** | **.411** | **.401** | **.398** | **.406** |

- Treino de 3 seeds e gate: 54 minutos (12:27 a 13:21). O robô puxou e
  carregou o modelo às 13:43, sem reiniciar, e roda a inferência a 2 Hz.
- O seed 0 passa em pallet_truck por margem mínima: o IC termina em −0,0097,
  e o limite é −0,010.
- iw_hub separado por teste, no modelo promovido: **.968** no laboratório
  (mesma sessão do treino) e **.277** no Roboflow (independente). O número
  que indica generalização é .277.

## Parte 2. Pergunta científica, testes e conclusão

Evidência **firme**: repete nos seeds ou tem IC claro. Evidência **limitada**:
amostra pequena, um único seed ou confundidor.

### 1. Qual modelo zero-shot serve como anotador automático? (limitada)

**Testes:** Grounding DINO tiny, MM-Grounding-DINO large, OWLv2 e SAM3 no
LOCO, com ajuste de prompts, em 150 imagens; SAM3 também no val completo.

**Conclusão:** SAM3 foi escolhido: AP50 .41–.48 em 150 imagens contra ~.14 do
OWLv2, ~.06 do Grounding DINO tiny e ~.01 do MM-Grounding-DINO. No val
completo, .30. O SAM3 recebeu mais rodadas de ajuste de prompts que os
outros, e o ensemble de prompts não superou o melhor prompt único.

### 2. Qual a qualidade dos rótulos do SAM3, e onde ele falha? (firme)

**Testes:** comparação com rótulos humanos em 2538 imagens do LOCO train (IoU
0,5); iw_hub em 509 imagens do Roboflow e falsos alarmes em 500 do LOCO; busca
de caixas que juntam pallets.

**Conclusão:** excelente em objeto único e distinto: iw_hub com precisão .99,
recall .73 e zero falsos alarmes. Fraco em cenas densas: precisão .31–.58 e
recall .26–.56 nas classes do LOCO. Em pallet, recall .26: 634 caixas juntam
pallets empilhados ou lado a lado, em 21% das imagens com 4 ou mais pallets.

### 3. Um detector treinado sem rótulo humano se aproxima do treinado com rótulos humanos? (firme)

**Testes:** YOLOv8m treinado do zero, 3 seeds, mesmas imagens, rótulos do LOCO
contra rótulos do SAM3 a 0,5, teste no LOCO val.

**Conclusão:** chega a 88% (.295 contra .337). SAM3 ganha em SLC (.427 contra
.349) e pallet_truck (.213 contra .173), empata em stillage e perde em
forklift (.078 contra .125). Pallet explica quase toda a perda (.319 contra
.616). Sem pallet, o SAM3 fica à frente (.289 contra .267).

### 4. As técnicas genéricas evitam o esquecimento ao adicionar uma classe? (limitada)

**Testes:** congelamento de BatchNorm, replay estratificado, destilação, peso
por classe, copy-paste, menos camadas congeladas, máscara de gradiente, treino
conjunto e YOLOv8l. Um seed por candidato.

**Conclusão:** nenhuma resolve: forklift cai ~30% e pallet_truck de 16% a 26%.
A máscara de gradiente piora tudo. Até o treino conjunto regride essas
classes, o que sugere que o problema não é só falta de replay.

### 5. Qual receita de modelo único preserva melhor as classes antigas? (firme)

**Testes:** oito variantes sob o mesmo protocolo, 3 seeds: ingênuo, replay,
destilação, copy-paste, self-training, banco híbrido, pseudo-rótulos no LOCO
train e treino conjunto. Antes disso: rodadas de self-training com aluno e
produção como professor, e limiares adaptativos.

**Conclusão:** a receita g (pseudo-rótulos no LOCO train) perde só 3,8%, contra
75% do ingênuo, 12% do replay e 9% do treino conjunto. Usar o aluno como
professor amplifica o esquecimento; o professor precisa ser o modelo que não
esqueceu. Limiares adaptativos pioraram e foram descartados. Nesse teste,
nenhuma variante passou no gate.

### 6. O esquecimento vem de rótulos ruins? (firme)

**Testes:** oráculo: mesmas 1760 imagens e mesma receita, trocando
pseudo-rótulos por rótulos humanos em stillage e pallet_truck, e depois em
todas as classes. 3 seeds.

**Conclusão:** não. Rótulos humanos não reduzem o esquecimento, e pallet_truck
piora (.261 para .222). Interpretação proposta: pseudo-rótulos do modelo
antigo agem como destilação. É o achado mais original, mas o mecanismo foi
inferido, não testado diretamente.

### 7. Isolar a classe nova num segundo modelo evita o esquecimento? (firme)

**Testes:** modelo LOCO intacto mais especialista YOLOv8s, com gate de falsos
positivos, 3 seeds.

**Conclusão:** evita por construção, a +19 ms por frame. Passa em 2 de 3 seeds;
um seed reprova por falsos positivos (2,06% contra 1%). Foi descartado em
07/10 pela decisão de manter um modelo único.

### 8. Até quantas classes o modelo único aguenta? (limitada)

**Testes:** adição incremental de até 70 classes do COCO, 300 imagens de treino
por classe.

**Conclusão:** o primeiro grupo perde 29% e estabiliza entre 40 e 70 classes; o
LOCO cai de .411 para .364 com 40 classes. O AP absoluto é baixo (11 das 70
classes abaixo de .01), então o teste mede tendência, não um modelo bom em
cada classe.

### 9. Como decidir com confiabilidade se um modelo pode ser implantado? (firme)

**Testes:** auditoria da avaliação, que achou três bugs: filtro de confiança
antes do AP, anotação com id 0 e teste influenciando a escolha do checkpoint.
Depois, gate com bootstrap pareado e IC de 95%, e leak check.

**Conclusão:** tolerância fixa é menor que o ruído entre seeds; o IC resolve
isso. Dois vazamentos reais apareceram: um híbrido que dava falso +19% por
usar armazéns do teste, que motivou o leak check, e 9 imagens de teste de
iw_hub aprovadas para o braço, que o leak check barrou.

### 10. O sistema aprende uma segunda classe sem esquecer a primeira? (limitada)

**Testes:** braço KUKA a partir de frames já gravados, 3 seeds, primeiro offline
e depois na rodada real, com teste de 84 imagens corrigidas à mão.

**Conclusão:** offline: braço .305, iw_hub +10%, pallet_truck −13%, 0 de 3 no
gate. Rodada real: braço .52–.56, 2 de 3 no gate. As duas configurações mudam
em várias coisas, então a causa do ganho não foi isolada. O teste vem do mesmo
ambiente do treino e foi pré-selecionado pelo SAM3.

### 11. O ciclo completo funciona de ponta a ponta no robô real? (limitada)

**Testes:** rodada real em 02/11: gravação por teleoperação, SAM3, revisão,
treino, gate, promoção, download pelo robô e inferência ao vivo.

**Conclusão:** sim, uma vez. Foi a primeira promoção desde o modelo inicial, com
margem mínima em pallet_truck e 2 de 3 seeds; a repetição da mesma
configuração passou em 1 de 3. Mostra viabilidade do sistema, não estabilidade
ao longo de várias rodadas.

### 12. Quanto a revisão humana agrega para uma classe nova? (limitada)

**Testes:** mesma rodada treinada com os frames aprovados pelo humano (107) e
com todos os que o SAM3 passou (140), 3 seeds, IC pareado.

**Conclusão:** nenhum ganho mensurável no braço (.542 contra .555, 0 de 3
seeds). Nas classes antigas o quadro muda: sem revisão, 0 de 3 modelos passam
no gate, todos por pallet_truck, contra 3 de 6 com revisão. A revisão não
melhorou a classe nova, mas pode proteger as antigas; a amostra é pequena, e
os braços diferem também no número de frames. Vale para revisão que só filtra
frames e para este teste, pré-selecionado pelo SAM3.

### 13. Imagens sintéticas melhoram a detecção em imagens reais? (firme)

**Testes:** rótulos humanos do LOCO mais 202 renders do Isaac Sim (7,4% do
treino), 3 seeds, validação e teste só com imagens reais. Uma primeira rodada
perdeu os rótulos de pallet e foi refeita.

**Conclusão:** sim: +12,8% (.337 para .380). Forklift +77%, pallet_truck +25% e
stillage +10%, as duas últimas em 3 de 3 seeds; nenhuma classe piora de forma
consistente. Vale para treino do zero e uma única proporção. Dentro do
pipeline contínuo ainda não foi testado.

### 14. O aprendido no robô generaliza para fora do ambiente? (firme)

**Testes:** iw_hub em 30 frames do laboratório contra 50 imagens independentes
do Roboflow; híbrido com imagens de armazéns de treino contra armazéns de
teste; val contra teste no treino do zero.

**Conclusão:** pouco: iw_hub cai de .968 para .277, e o treino do zero cai de
~.77 de mAP50 no val para ~.30–.38 no teste. A diferença entre ambientes é o
principal limite do sistema.

### 15. Um modelo de linguagem com visão substitui o anotador? (limitada)

**Testes:** Claude Opus 5.5 anotando à mão 26 imagens do LOCO val, com grade de
coordenadas. Teste interrompido antes das 100 imagens planejadas.

**Conclusão:** não para cenas densas: em pallet, precisão .38 e recall .09.
Pallet_truck vai bem, mas com só 5 caixas. As 6 imagens usadas para aprender as
convenções estão incluídas nesses números. O uso como revisor de caixas não
foi testado.

### 16. A receita funciona em um benchmark externo? (limitada)

**Testes:** CUB, protocolo de Incremental Zero-Shot Learning de Wei et al.
(2022), sem replay, com 5 imagens por classe e com replay completo.

**Conclusão:** com replay completo, H = 45,2 contra 41,5 do paper; sem replay,
19,5. O método depende de guardar dados antigos, o que o paper não faz, e o
pré-treino do SAM3 pode incluir essas espécies.

## Limitações e pendências

- Só 3 seeds, um único ambiente de robô e uma única rodada real.
- A aprovação no gate está perto do limite e varia entre repetições.
- Teste do braço: 84 imagens do mesmo ambiente, pré-selecionadas pelo SAM3, sem
  negativos.
- Teste independente do iw_hub: 50 imagens do Roboflow. A gravação de 02/11 à
  tarde segue sem rótulos.
- Muitos frames quase duplicados: na gravação da tarde, 391 viraram 88.
- Qualidade do SAM3: `sam3_label_quality.json` traz números diferentes dos
  usados aqui para as classes do LOCO (por exemplo SLC .59 / .65). É preciso
  conferir a diferença de amostra antes de citar na tese.
