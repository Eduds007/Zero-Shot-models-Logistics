# Esquecimento catastrófico ao adicionar `iw_hub`: investigação completa

Registro de todas as tentativas de evitar que o fine-tuning incremental do
pipeline (`dataset/pipeline/`) esqueça as 5 classes originais do LOCO
(`small_load_carrier`, `forklift`, `pallet`, `stillage`, `pallet_truck`) ao
aprender a nova classe `iw_hub` ("robô amarelo"). Cada rodada reinicia do
checkpoint LOCO-puro (`runs/detect/yolo_runs/loco_full_v3/weights/best.pt`),
com a maior parte da rede congelada (`freeze=N`, default 21) e um replay do
LOCO misturado ao batch de fine-tune -- ver o docstring de
`build_training_batch.py` para o desenho completo.

## Metodologia de avaliação

Duas fontes de avaliação foram usadas ao longo da investigação, e é
importante não confundi-las:

1. **Gate oficial** (`promote_if_better.py`): val oficial do LOCO (2277
   imagens fixas, `dataset/loco_val_paths.txt`) + held-out do Roboflow para
   `iw_hub` (509 imagens, `dataset/pipeline/held_out_val/iw_hub/`). Essa é a
   referência primária -- amostra grande, estatisticamente mais confiável.
2. **Test set fixo** (`build_fixed_testset.py` + `eval_fixed_testset.py`,
   construído nesta investigação): 200 imagens estratificadas do split
   oficial do LOCO + 30 imagens reais do vídeo do laboratório (carve-out
   permanente do pool de treino aprovado) + 50 imagens do Roboflow. Criado
   pra fechar uma lacuna real: as validações internas de cada rodada de
   `run_finetune.py` vinham de uma amostra do próprio batch de treino,
   reamostrada a cada rodada (bug real encontrado e corrigido: a consulta
   SQL de `db.approved_pool()` não tinha `ORDER BY`, então o seed fixo do
   `random.shuffle()` não garantia a mesma ordem entre execuções). Com o
   test set fixo, o mesmo conjunto de imagens é usado pra sempre, mesmo
   depois de sucessivas rodadas de treino.

**Achado à parte**: comparar `iw_hub` no vídeo real do laboratório (0.91-0.94
AP_50) contra o held-out do Roboflow (0.14-0.30 AP_50) revela um gap de
generalização enorme -- o modelo aprende muito bem as condições específicas
do laboratório, mas não generaliza bem pra outros domínios/ângulos/fundos.

## Bugs de engenharia corrigidos no caminho

- `db.approved_pool()` sem `ORDER BY` -- amostragem com seed fixo não era
  reprodutível entre execuções (ver acima).
- `promote_if_better.py`: `known_classes()` nunca incluía uma classe nova
  (`iw_hub`) até ela já ter sido promovida uma vez -- corrigido pra sempre
  unir as classes de produção com as classes `source=pipeline` do registry.
- `promote_if_better.py`: avaliar o modelo de produção (que não conhece
  `iw_hub`) contra o held-out de `iw_hub` sempre dava AP_50=0.0, e o gate
  tratava isso como "baseline real" em vez de aplicar o piso mínimo de
  primeira promoção -- corrigido pra só avaliar produção nas classes que
  ela realmente conhece.

## Padrão de avaliação: por que "retenção relativa" em vez de tolerância absoluta

O gate de produção usa uma tolerância absoluta fixa (`REGRESSION_TOLERANCE
= -0.01`). Para julgar as técnicas abaixo de forma mais justa, medimos
também a **retenção relativa** de cada classe (candidato ÷ produção), por
dois motivos observados nesta própria investigação:

1. Batches de inferência com composição diferente (mesmo modelo, mesmas
   imagens) já produziram ~0.028 de diferença em AP_50 só por ruído
   numérico -- maior que a tolerância absoluta de -0.01.
2. Uma tolerância absoluta pune desproporcionalmente classes com baseline
   baixo (`pallet_truck`, AP_50=0.086): uma queda de 0.01 já é ~12%
   relativo, contra ~2% relativo pra uma classe como `pallet` (0.427).

Padrão usado nas tabelas abaixo: **retenção relativa >= 90%** (queda
relativa máxima de 10%) por classe, sobre o gate oficial.

## Todas as técnicas testadas

1. **Congelamento de BatchNorm** -- verificação de código. O ultralytics
   instalado (8.4.131) já congela `.eval()` corretamente nas camadas
   congeladas via `freeze=N`. Não era a causa.
2. **Amostragem estratificada do replay LOCO** (`build_training_batch.py`,
   `_stratified_loco_sample`) -- garante um mínimo de imagens por classe
   LOCO (raras primeiro) em vez de amostragem puramente aleatória. Piso
   agressivo piorou (inflou o ratio efetivo e diluiu `iw_hub`); piso
   moderado não trouxe ganho além da distillation.
3. **Knowledge distillation nativa do ultralytics** (`distill_model=`,
   `run_finetune.py`) -- distillation de features do neck ponderada pela
   confiança do professor (modelo de produção). Ajuda um pouco nas 3
   classes LOCO mais afetadas, mas sozinha não fecha o gap; piora `iw_hub`.
4. **`cls_pw`** (peso de classe por frequência inversa, nativo do
   ultralytics) -- efeito parecido com a distillation, mesma limitação.
5. **Copy-paste augmentation para `iw_hub`** (`copy_paste_augment.py`) --
   recorta cada instância de `iw_hub` via bbox e cola em fundos do LOCO
   (3 pastes por fonte). Melhor resultado isolado antes desta rodada de
   testes: `iw_hub` dispara, `pallet_truck` chega a empatar com produção.
6. **Reduzir `freeze` (21->15)** -- mais capacidade destravada. Único teste
   isolado (antes desta rodada) a cruzar o piso mínimo de `iw_hub`.
7. **Máscara de gradiente nas classes LOCO** ("dual-head" pobre, hook nativo
   do PyTorch `register_hook()` zerando o gradiente dos canais LOCO na
   cabeça de classificação) -- **piorou geral**. Sem poder se readaptar, o
   classificador LOCO fica desalinhado com as features compartilhadas que
   ainda mudam sob o único bloco destravado.
8. **Combinações** (estratificação + distillation; copy-paste + freeze=15)
   -- não somam os pontos fortes; tendem a competir pelo mesmo orçamento de
   capacidade treinável.
9. **Treino conjunto (diagnóstico)** -- em vez do replay proporcional,
   inclui quase todo o LOCO de treino (2820 imagens) + `iw_hub`, com
   `freeze=10` (muito mais capacidade destravada). Hipótese: eliminaria a
   disputa de capacidade quase por definição. **Refutada** -- `forklift`
   piorou ainda mais (-38% relativo, pior que qualquer teste anterior).
10. **Capacidade maior de modelo (YOLOv8l em vez de YOLOv8m)** -- novo
    baseline LOCO-puro em YOLOv8l (40 épocas) + mesma receita de copy-paste.
    `stillage` melhorou de verdade (+21.5%, o único caso da investigação
    inteira onde uma classe LOCO ficou **melhor** que a produção), mas
    `forklift` teve o pior resultado de toda a investigação (-46.7%
    relativo). Latência: 10.6ms/imagem vs. 7.2ms do v8m em produção (47%
    mais lento em termos relativos, mas folgado dentro do orçamento real do
    robô, que roda a 2Hz = 500ms por frame).

## Tabela final -- gate oficial, retenção relativa por classe

| classe | produção | v13 (copy-paste, v8m) | treino conjunto (v8m, freeze=10) | capacidade maior (v8l, copy-paste) |
|---|---|---|---|---|
| small_load_carrier | 0.205 | 0.204 (-0.5%) ✅ | 0.189 (-7.8%) ✅ | 0.208 (+1.5%) ✅ |
| **forklift** | 0.225 | 0.144 (-36.0%) ❌ | 0.139 (-38.2%) ❌ | 0.120 (-46.7%) ❌ |
| pallet | 0.427 | 0.426 (-0.2%) ✅ | 0.425 (-0.5%) ✅ | 0.408 (-4.4%) ✅ |
| stillage | 0.311 | 0.271 (-12.9%) ❌ | 0.291 (-6.4%) ✅ | 0.378 (+21.5%) ✅ |
| pallet_truck | 0.086 | 0.077 (-10.5%) ❌ | 0.066 (-23.3%) ❌ | 0.063 (-26.7%) ❌ |
| iw_hub (gate oficial) | -- | 0.207 | 0.312 | 0.159 |

## Conclusão

Depois de ~17 variações testadas ao longo de toda a investigação, nenhuma
isolada ou combinada resolveu o esquecimento por completo -- mas o quadro
mudou de "esquecimento generalizado" para um problema muito mais
específico: **`forklift` nunca ficou dentro de nenhuma tolerância razoável,
em NENHUMA técnica testada** (sempre entre -29% e -47% relativo,
independente do mecanismo). Em contraste, `stillage`, `pallet` e
`small_load_carrier` já passam (ou chegam perto) em pelo menos uma das
técnicas mais recentes. Isso sugere fortemente que o próximo passo não é
mais testar técnicas genéricas de anti-forgetting -- é investigar
especificamente por que `forklift` é tão mais frágil que as outras 4
classes (representação visual mais próxima de `iw_hub`? menos exemplos de
treino? maior variância intra-classe nas poses do LOCO?).

Nenhum dos candidatos aqui foi promovido a produção -- `promote_if_better.py`
corretamente bloqueou todos por causa de `forklift`.
