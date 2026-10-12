# Esquecimento catastrófico ao adicionar `iw_hub`: investigação completa

Registro de todas as tentativas de evitar que o fine-tuning incremental do
pipeline (`dataset/pipeline/`) esqueça as 5 classes originais do LOCO
(`small_load_carrier`, `forklift`, `pallet`, `stillage`, `pallet_truck`) ao
aprender a nova classe `iw_hub` ("robô amarelo"). Cada rodada reinicia do
checkpoint LOCO-puro (`runs/detect/yolo_runs/loco_full_v3/weights/best.pt`),
com a maior parte da rede congelada (`freeze=N`, default 21) e um replay do
LOCO misturado ao batch de fine-tune -- ver o docstring de
`build_training_batch.py` para o desenho completo.

> **Correções em relação à primeira versão deste documento.** Os números da
> primeira versão estavam errados por dois bugs de avaliação (ver "Bugs de
> avaliação" abaixo): AP_50 subestimado, sobretudo nas classes fracas, e
> `iw_hub` no vídeo do laboratório reportado como 0.91-0.94 quando o valor
> real é ~1.0. A tabela de técnicas foi reavaliada com a métrica corrigida, e
> as conclusões sobre `forklift` mudaram. As seções a partir de "Sistema de
> dois modelos" registram os experimentos feitos depois.

## Metodologia de avaliação

1. **Classes do LOCO**: val oficial do LOCO (2277 imagens fixas,
   `dataset/loco_val_paths.txt`). O split oficial do LOCO é **por armazém**:
   treino = subsets 2, 3 e 5; val = subsets 1 e 4. Ou seja, a val mede
   generalização para armazéns nunca vistos.
2. **`iw_hub`**: test set fixo (`build_fixed_testset.py` +
   `eval_fixed_testset.py`): 30 frames reais do vídeo do laboratório
   (carve-out permanente do pool de treino) + 50 imagens do Roboflow. O
   held-out antigo de 509 imagens do Roboflow
   (`dataset/pipeline/held_out_val/iw_hub/`) era feito de symlinks para
   `/tmp/iwhub_export`, apagado no reboot da workstation em 2026-10-01; o
   gate (`promote_if_better.py`) ainda aponta para essa pasta e hoje avalia
   `iw_hub` com 0 imagens (AP_50 = 0.000) -- pendente de corrigir.
3. **Ruído entre seeds**: o mesmo treino com seeds diferentes varia até
   ~±0.05 de AP_50 nas classes fracas (`forklift`, `pallet_truck`). Desde o
   self-training, toda comparação usa **3 seeds** e batch fixo (8) -- o
   AutoBatch (`batch=-1`) escolhia de 4 a 16 conforme a GPU compartilhada
   estava livre, o que confundia as comparações.

**Gap de generalização de `iw_hub`**: ~1.0 de AP_50 no vídeo do laboratório
contra ~0.3-0.4 no Roboflow -- o modelo aprende as condições do laboratório,
mas não generaliza para outros fundos e ângulos.

## Bugs de avaliação (corrigidos)

- **Filtro de confiança antes do AP**: `predict_yolo()` usava `conf=0.5`
  antes do COCOeval, cortando a curva precisão-recall. Corrigido para
  `AP_CONFIDENCE_THRESHOLD = 0.001`. Produção com a métrica corrigida:
  `small_load_carrier` 0.385, `forklift` 0.262, `pallet` 0.632, `stillage`
  0.496, `pallet_truck` 0.278 (média 0.411).
- **Anotação com id 0**: o COCOeval guarda o id da anotação casada e trata 0
  como "não casada", então a primeira caixa de cada conjunto nunca contava
  como acerto. Corrigido começando os ids em 1 (era o que derrubava `iw_hub`
  de ~1.0 para 0.91-0.94).
- **Vazamento na escolha do checkpoint**: o split `val` dos batches de
  fine-tune, usado pelo ultralytics para escolher o `best.pt` e para a
  parada antecipada, continha 170 imagens da val do LOCO -- as mesmas do
  teste. O modelo não treinava nelas, mas o teste influenciava qual época
  era escolhida. Afeta todos os experimentos até o self-training (inclusive).
  Os experimentos limpos treinam 30 épocas fixas e avaliam o `last.pt`.

## Bugs de engenharia corrigidos no caminho

- `db.approved_pool()` sem `ORDER BY` -- amostragem com seed fixo não era
  reprodutível entre execuções.
- `promote_if_better.py`: `known_classes()` nunca incluía uma classe nova
  (`iw_hub`) até ela já ter sido promovida uma vez -- corrigido pra sempre
  unir as classes de produção com as classes `source=pipeline` do registry.
- `promote_if_better.py`: avaliar o modelo de produção (que não conhece
  `iw_hub`) contra o held-out de `iw_hub` sempre dava AP_50=0.0, e o gate
  tratava isso como "baseline real" em vez de aplicar o piso mínimo de
  primeira promoção -- corrigido pra só avaliar produção nas classes que
  ela realmente conhece.

## Técnicas genéricas de anti-forgetting

1. **Congelamento de BatchNorm** -- verificação de código. O ultralytics
   instalado (8.4.131) já congela `.eval()` corretamente nas camadas
   congeladas via `freeze=N`. Não era a causa.
2. **Amostragem estratificada do replay LOCO** (`build_training_batch.py`,
   `_stratified_loco_sample`) -- garante um mínimo de imagens por classe
   LOCO (raras primeiro). Piso agressivo piorou (inflou o ratio efetivo e
   diluiu `iw_hub`); piso moderado não trouxe ganho além da distillation.
3. **Knowledge distillation nativa do ultralytics** (`distill_model=`,
   `run_finetune.py`) -- distillation de features do neck ponderada pela
   confiança do professor. Ajuda pouco; sozinha não fecha o gap.
4. **`cls_pw`** (peso de classe por frequência inversa) -- efeito parecido
   com a distillation, mesma limitação.
5. **Copy-paste augmentation para `iw_hub`** (`copy_paste_augment.py`) --
   recorta cada instância de `iw_hub` e cola em fundos do LOCO.
6. **Reduzir `freeze` (21->15)** -- mais capacidade destravada.
7. **Máscara de gradiente nas classes LOCO** (hook `register_hook()`
   zerando o gradiente dos canais LOCO na cabeça de classificação) --
   **piorou geral**: o classificador LOCO fica desalinhado com as features
   compartilhadas, que continuam mudando.
8. **Combinações** (estratificação + distillation; copy-paste + freeze=15)
   -- não somam os pontos fortes.
9. **Treino conjunto (diagnóstico)** -- quase todo o LOCO de treino (2820
   imagens) + `iw_hub`, com `freeze=10`.
10. **YOLOv8l em vez de YOLOv8m** -- novo baseline LOCO-puro em YOLOv8l (40
    épocas) + copy-paste. Latência: 10.6 ms/imagem contra 7.2 ms do v8m na
    workstation.

### Tabela -- métrica corrigida (AP_50 na val oficial do LOCO)

Uma seed por candidato; diferenças menores que ~0.05 nas classes fracas
estão dentro do ruído entre seeds.

| classe | produção (v8m) | v13 (copy-paste, v8m) | treino conjunto (v8m, freeze=10) | YOLOv8l (copy-paste) |
|---|---|---|---|---|
| small_load_carrier | 0.385 | 0.376 (−2%) | 0.365 (−5%) | 0.451 (+17%) |
| **forklift** | 0.262 | 0.183 (−30%) | 0.175 (−33%) | 0.179 (−32%) |
| pallet | 0.632 | 0.637 (+1%) | 0.631 (0%) | 0.637 (+1%) |
| stillage | 0.496 | 0.459 (−7%) | 0.474 (−4%) | 0.556 (+12%) |
| **pallet_truck** | 0.278 | 0.206 (−26%) | 0.211 (−24%) | 0.233 (−16%) |
| iw_hub (lab 30 / Roboflow 50) | -- | 0.997 / 0.421 | 0.990 / 0.338 | 1.000 / 0.311 |

Com a métrica corrigida, todas as técnicas genéricas regridem `forklift`
(−30% a −33%) e `pallet_truck` (−16% a −26%); o treino conjunto, com
praticamente todo o LOCO anotado, também. O YOLOv8l melhora
`small_load_carrier` e `stillage`, mas a comparação é contra a produção em
v8m. A primeira versão deste documento dizia que o YOLOv8l era o pior
candidato -- era efeito do filtro de confiança 0.5.

## Sistema de dois modelos (`two_model_gate.py`, `train_specialist.py`)

Modelo LOCO congelado (o de produção, intocado) + um YOLOv8s especialista
só para as classes novas do pipeline; as caixas do especialista entram onde
ele tem confiança >= 0.5 e IoU > 0.7 com caixas do LOCO. É isolamento de
parâmetros (como Progressive Nets).

- Classes do LOCO: **idênticas à produção** (só 1 caixa suprimida, com
  confiança < 0.5).
- Falsos positivos do especialista: 0.26% das imagens do LOCO (limite: 1%).
- `iw_hub`: estatisticamente equivalente ao melhor modelo único
  (bootstrap pareado, 95%).
- Custo no robô: 57 ms por frame contra 38 ms do v8m sozinho (11% do
  orçamento de 500 ms a 2 Hz).

**Passou no gate. Não foi promovido nem integrado ao robô** (decisão
pendente).

## Teste de saturação (`saturation_test.py`)

Quantas classes o modelo único (receita incremental, sem especialista)
aprende até a perda das classes antigas parar de crescer. Classes do COCO
(sem `person`), 300 imagens de treino e 200 de avaliação por classe, passos
de 1, 2, 5, 10, 20, 40 e 70 classes. AP_50 médio de cada grupo de classes:

| grupo | logo após aprender | 20 classes | 40 classes | 70 classes | perda total |
|---|---|---|---|---|---|
| 1-10 | 0.112 | 0.098 | 0.082 | 0.079 | −29% |
| 11-20 | 0.123 | — | 0.116 | 0.103 | −16% |
| 21-40 | 0.078 | — | — | 0.072 | −7% |

A queda das 10 primeiras quase para entre 40 e 70 classes (0.082 -> 0.079):
é onde aparece a saturação. LOCO: 0.411 -> 0.364 com 40 classes (−11%).
Limitação: o AP absoluto das classes novas é baixo (11 das 70 abaixo de
0.01), então o teste mede a tendência do esquecimento, não um modelo que
aprendeu bem cada classe. (A linha de "platô" em `saturation/SUMMARY.md`
compara conjuntos de classes diferentes e está errada; os números acima são
por grupo fixo.)

## Self-training (`self_training.py`, `pseudo_label_old_classes.py`)

Ideia do orientador, na linha de Mayershofer, Fischer e Fottner, "Improving
visual object detection using synthetic self-training" (Logistics Journal,
2021): um modelo professor rotula as classes antigas nos frames novos, e o
treino se repete por rodadas.

- **Vídeo de celular (79 frames)**: quase não tinha forklifts nem pallet
  trucks; a comparação com 3 seeds mostrou diferenças dentro do ruído.
- **390 frames da câmera do robô (2026-10-05), aluno como professor**:
  `forklift` 0.206 -> 0.252, mas `pallet_truck` 0.220 -> 0.190 -- o aluno,
  que já esquecia `pallet_truck`, rotulava menos a cada rodada (55 -> 30 ->
  19 rótulos) e amplificava o esquecimento.
- **Mesmos frames, produção como professor das classes do LOCO em todas as
  rodadas**: `forklift` 0.256 (produção 0.262), `pallet_truck` 0.216 --
  `forklift` recuperado sem piorar nenhuma classe. As rodadas extras não
  acrescentaram nada além da primeira: o ganho vem dos frames do próprio
  ambiente rotulados por um modelo que não esqueceu.

## Imagens não anotadas do LOCO com pseudo-rótulos

5.000 das ~32 mil imagens não anotadas do LOCO
(`sam3_label_loco_unannotated.py`), somadas aos 390 frames do robô.

- **Só SAM3 nas 5 classes**: o modelo fica tão bom quanto o anotador --
  `forklift` (SAM3 zero-shot: AP_50 0.012) e `pallet` desabam.
- **Híbrido** (`hybrid_relabel_sam3_loco.py`): SAM3 em `small_load_carrier`,
  `stillage` e `pallet_truck` (onde ele é melhor que a produção) e produção
  em `forklift` e `pallet`.
- **Vazamento**: a primeira amostra sorteou 1.000 imagens de cada um dos 5
  armazéns, incluindo os 2 de validação (15.7% das imagens a menos de 1 s de
  um frame de teste). Com isso o híbrido deu média 0.491 (+19% sobre a
  produção) -- resultado inválido. Refeito só com os armazéns 2, 3 e 5, com
  checagem automática de vazamento (`train_with_sam3_loco.py`) e 30 épocas
  fixas:

| classe | produção | só frames do robô | híbrido contaminado | **híbrido limpo** (3 seeds) | só SAM3, limpo |
|---|---|---|---|---|---|
| forklift | 0.262 | 0.256 | 0.306 | **0.253** | 0.152 |
| pallet | 0.632 | 0.635 | 0.630 | **0.633** | 0.464 |
| stillage | 0.496 | 0.468 | 0.578 | **0.465** | 0.467 |
| small_load_carrier | 0.385 | 0.369 | 0.513 | **0.374** | 0.366 |
| pallet_truck | 0.278 | 0.216 | 0.429 | **0.242** | 0.242 |
| média LOCO | 0.411 | 0.389 | 0.491 | **0.393** | 0.338 |

O mesmo método com imagens do mesmo ambiente do teste deu +19%; com imagens
de outros armazéns, quase nada. O limite principal parece ser a **diferença
entre ambientes**, não só a quantidade de exemplos.

## Gate com 95% de confiança (`promote_if_better.py`)

A tolerância fixa de −0.01 é menor que o ruído entre seeds. O gate agora só
reprova uma classe se o intervalo de 95% de confiança da diferença
candidato − produção (bootstrap pareado sobre as mesmas imagens, 1000
reamostragens) ficar **inteiro** abaixo de −0.01. `--dry-run` avalia sem
promover. Híbrido limpo, 3 seeds:

| classe | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| small_load_carrier | passa | passa | passa |
| forklift | passa | passa | passa |
| pallet | passa | passa | passa |
| stillage | reprova [−0.055, −0.019] | passa [−0.040, −0.003] | reprova [−0.056, −0.021] |
| pallet_truck | reprova [−0.054, −0.012] | reprova [−0.058, −0.016] | reprova [−0.064, −0.020] |

## Situação atual

- **Modelo único**: `forklift`, `pallet` e `small_load_carrier` já ficam no
  nível da produção (dentro do ruído). **`pallet_truck` (−13%) e `stillage`
  (−6%) continuam comprovadamente piores** -- nenhum modelo único passou no
  gate.
- **Sistema de dois modelos**: passou no gate, ao custo de ~19 ms a mais por
  frame no robô.
- **Limiares adaptativos por classe** (`apple_threshold_relabel.py`), na
  linha de Song et al., "Overcoming Catastrophic Forgetting for Multi-Label
  Class-Incremental Learning" (APPLE, WACV 2024): o limiar de cada classe é
  escolhido para que os pseudo-rótulos tenham tantos objetos por imagem
  quanto o LOCO anotado (com 0.5 fixo, `pallet_truck` tinha 0.32 objetos por
  imagem contra 0.59 no LOCO, e `stillage` 0.82 contra 1.21). Resultado (3
  seeds): `pallet_truck` 0.242 -> 0.252 e `pallet` 0.633 -> 0.642, mas
  `stillage` 0.465 -> 0.446 e `forklift` 0.253 -> 0.214 -- os rótulos extras
  de baixa confiança são ruidosos. Média LOCO 0.386 (pior que o híbrido) e
  `stillage` reprova nas 3 seeds. **Descartado.**

## Decisão final (2026-10-07)

**Modelo único**, com a receita do híbrido limpo e **pseudo-rótulos sempre
com confiança 0.5** (sem limiar adaptativo, sem especialista). A receita
está implementada no pipeline (`build_training_batch.py`, `pseudo_labels.py`,
ver `README.md`). Uma rodada completa do pipeline (3 seeds) reproduziu o
resultado: média LOCO 0.385, `stillage` e `pallet_truck` ainda reprovam no
gate, então nada é promovido até essas duas classes melhorarem.

## Experimento à parte: Incremental Zero-Shot Learning no CUB

`dataset/izsl_cub/cub_izsl.py`: a receita no benchmark CUB de Wei et al.,
"Incremental Zero-Shot Learning" (IEEE T-Cybernetics 2022), Tabela II
(split padrão de ZSL, 5 etapas de 30 classes, SAM3 para as 50 classes não
vistas, portão de confiança 0.5 do YOLO):

| método | guarda imagens antigas? | u | s | H |
|---|---|---|---|---|
| Wei et al. | não | 38.1 | 45.6 | 41.5 |
| nosso, sem replay | não | 33.4 | 13.7 | 19.5 |
| nosso, 5 imagens por classe | 5/classe | 44.7 | 26.0 | 32.9 |
| nosso, replay do pipeline (todas) | todas | 44.1 | 46.3 | 45.2 |
| treino conjunto (teto do paper) | todas | 43.7 | 57.7 | 49.7 |

Sem replay o método colapsa (cada rodada recomeça do modelo base, então as
classes antigas só sobrevivem pelo replay); o forte é o SAM3 (73.8% sozinho
nas classes não vistas, contra 51.2% do paper -- com a ressalva de que o
pré-treino do SAM3 pode incluir essas espécies). O método serve para o
cenário em que o dataset antigo continua disponível, não para o IZSL sem
armazenamento.
