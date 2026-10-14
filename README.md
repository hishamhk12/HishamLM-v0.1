# HishamLM

نموذج لغوي (Language Model) نبنيه من الصفر: أوزان تبدأ عشوائيًا، architecture مكتوبة يدويًا،
بدون أي pretrained model أو مكتبة tokenizer أو Hugging Face. PyTorch مستخدم فقط للـ tensors،
الطبقات الأساسية (`Linear`, `Embedding`, `LayerNorm`, `Dropout`)، autograd والـ optimizer.

**الحالة الحالية: المرحلة الثانية** — أول Transformer Language Model مكتوب يدويًا (self-attention بخطواتها كاملة)، مع الإبقاء على MLP المرحلة الأولى للمقارنة.

## بنية المشروع

```
HishamLM/
├── configs/
│   ├── mlp_char.json             إعدادات MLP (المرحلة الأولى)
│   └── transformer_char.json     إعدادات Transformer
├── data/
│   ├── raw/                      ملفات .txt للتدريب (فيها sample صغير للاختبار)
│   └── processed/                محجوز للمراحل القادمة
├── tokenizer/char_tokenizer.py   Character-level tokenizer
├── model/
│   ├── mlp_lm.py                 MLP Language Model
│   ├── attention.py              Causal multi-head self-attention
│   ├── transformer_lm.py         FeedForward، TransformerBlock، TransformerLanguageModel
│   └── build.py                  build_model(): يختار الموديل حسب model_type
├── training/
│   ├── config.py                 TrainConfig (dataclass يُقرأ من JSON)
│   ├── data.py                   قراءة النصوص، split، batching
│   ├── utils.py                  seed و device
│   └── train.py                  training loop (للموديلين)
├── inference/generate.py         توليد نص token-by-token (للموديلين)
├── checkpoints/                  مخرجات التدريب (غير مرفوعة على git)
├── tests/                        unit tests
└── requirements.txt
```

## كيف يعمل

### 1. Tokenizer (`tokenizer/char_tokenizer.py`)
- كل حرف = token واحد.
- الـ vocabulary تُبنى من كل الحروف المختلفة في النص (مرتبة).
- id رقم `0` محجوز لـ `<unk>`: أي حرف غير موجود في الـ vocabulary يتحول إليه في `encode()`.
- `stoi` (char → id)، `itos` (id → char)، `encode()`، `decode()`، `save()`/`load()` بصيغة JSON.

### 2. Dataset pipeline (`training/data.py`)
1. قراءة كل ملفات `data/raw/*.txt` بترتيب الاسم ودمجها بسطر جديد.
2. تحويل النص كاملًا إلى token IDs.
3. أول 90% للتدريب وآخر 10% للـ validation (`val_fraction`).
4. `get_batch()` يختار مواضع عشوائية ويُرجع:
   - `x`: نافذة بطول `context_length`
   - `y`: نفس النافذة مُزاحة بحرف واحد، أي `y[t]` هو الحرف الذي يأتي بعد `x[t]`.

### 3. الموديلات
كلا الموديلين لهما نفس الـ interface: `(B, T) token ids → (B, T, vocab_size) logits`.
لذلك الـ data pipeline والـ training loop والـ generation مشتركة.

#### MLP (`model/mlp_lm.py`)
MLP Language Model (على فكرة Bengio 2003):

```
token ids (B, T)
  → Embedding                      (B, T, C)
  → لكل موضع t: ضم embeddings آخر n حروف (t-n+1 … t)   (B, T, n·C)
  → Linear + tanh                  (B, T, hidden)
  → Linear                         (B, T, vocab)
```

#### Transformer (`model/transformer_lm.py` + `model/attention.py`)
Decoder-only Transformer بأسلوب pre-norm:

```
token ids (B, T)
  → token embedding + learned positional embedding           (B, T, C)
  → n_layers × Block:
        x = x + Attention(LayerNorm(x))     ← الحروف تتبادل المعلومات
        x = x + FeedForward(LayerNorm(x))   ← كل حرف يُعالَج وحده
  → final LayerNorm
  → Linear LM head                                            (B, T, vocab)
```

**Causal Self-Attention** (`model/attention.py`) مكتوبة خطوة بخطوة:

```
Q = X·Wq,  K = X·Wk,  V = X·Wv                      (B, T, C)
تقسيم C إلى H heads بحجم D = C/H                      (B, H, T, D)
scores  = Q·Kᵀ / √D                                   (B, H, T, T)
scores[i, j] = -inf  لكل j > i   (causal mask بـ torch.tril)
weights = softmax(scores)    ← كل صف توزيع احتمالي على المواضع المسموحة
out     = weights · V                                 (B, H, T, D)
دمج الـ heads                                          (B, T, C)
out     = out·Wo   (output projection)
```

أي: `Attention(Q, K, V) = softmax(Q·Kᵀ/√D + M)·V` حيث `M[i,j] = 0` إذا `j ≤ i` و `-inf` إذا `j > i`.

قرارات التصميم:
- **Pre-norm** (`x + f(LN(x))`) بدل post-norm: الـ residual stream يمر بدون تعديل من البداية للنهاية، فالتدريب أكثر استقرارًا ولا يحتاج learning-rate warmup في الموديلات الصغيرة.
- **GELU** في الـ FeedForward (`C → 4C → C`): دالة ناعمة لا تقطع الـ gradient تمامًا للقيم السالبة، وهي المعيار في GPT-2. على هذا الحجم ReLU كانت ستعطي نتيجة مشابهة؛ الاختيار للتوافق مع الممارسة الشائعة وليس لفرق مُثبت هنا.
- **Initialization**: كل الأوزان عشوائية `Normal(0, 0.02)`، الـ biases أصفار، LayerNorm بقيمها الافتراضية (1 و 0). الطبقات التي تكتب في الـ residual stream (`W_o` و `fc_out`) تأخذ `std = 0.02/√(2·n_layers)` حتى لا يكبر الـ variance مع عدد الـ blocks. النتيجة: loss أولي ≈ `ln(vocab_size)`.
- `embedding_dim % n_heads != 0` يعطي `ValueError` واضح.
- Sequence أطول من `context_length` يعطي `ValueError` (جدول الـ positional embedding بهذا الطول فقط).

### 4. Training loop (`training/train.py`)
لكل خطوة: batch → forward → cross-entropy loss → `backward()` → gradient clipping → `optimizer.step()` (AdamW) → `zero_grad()`.

- قبل التدريب (step 0) وكل `eval_interval` خطوة: train/val loss على نفس الـ batches الثابتة كل مرة.
- `checkpoints/<run_name>/latest.pt`: يُحفظ في كل تقييم، للاستكمال.
- `checkpoints/<run_name>/best.pt`: يُحفظ فقط عندما يتحسن الـ val loss.
- الـ checkpoint يحتوي: `model_type`، أوزان الموديل، حالة الـ optimizer، الـ config، الـ vocabulary، رقم الخطوة، الـ losses، `best_val_loss`، وحالة الـ random generators.
- **Resume** (`--resume`): يحمّل `latest.pt` ويكمل حتى `max_steps`. لأننا نحفظ حالة الـ random generators (اختيار الـ batches + الـ dropout)، التدريب المُستكمَل يعطي نفس الأوزان بالضبط كأنه لم يتوقف (مُختبر).
- seed ثابت → نفس النتائج تمامًا عند إعادة التشغيل (مُختبر).
- أيضًا يُكتب `log.csv` و`vocab.json`.

### 5. Generation (`inference/generate.py`)
يقرأ `model_type` من الـ checkpoint ويبني الموديل المناسب. ثم في حلقة: آخر `context_length` token → الموديل → logits آخر موضع →
- `--temperature 0`: greedy، يختار أعلى احتمال مباشرة (deterministic).
- `--temperature > 0`: يقسم على temperature، (اختياريًا `--top-k`)، softmax، ثم sampling.

## التشغيل

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# الاختبارات
.venv/bin/python -m unittest discover tests -v

# التدريب
.venv/bin/python -m training.train --model transformer       # configs/transformer_char.json
.venv/bin/python -m training.train --model mlp               # configs/mlp_char.json
.venv/bin/python -m training.train --config configs/my.json  # أي config

# استكمال التدريب من latest.pt (مثلًا حتى 4000 خطوة)
.venv/bin/python -m training.train --model transformer --resume --max-steps 4000

# التوليد
.venv/bin/python -m inference.generate --checkpoint checkpoints/transformer_char/latest.pt --prompt "The cat"
.venv/bin/python -m inference.generate --checkpoint checkpoints/transformer_char/best.pt --prompt "القط" --top-k 10
.venv/bin/python -m inference.generate --checkpoint checkpoints/mlp_char/latest.pt --prompt "القط" --temperature 0
```

كل الأوامر تُشغَّل من جذر المشروع. لإضافة بيانات: ضع ملفات `.txt` (UTF-8) في `data/raw/`.

## الإعدادات

| المفتاح | المعنى |
|---|---|
| `run_name` | اسم مجلد الـ checkpoints |
| `model_type` | `mlp` أو `transformer` |
| `seed` | الـ random seed |
| `device` | `auto` أو `cpu` أو `cuda` |
| `val_fraction` | (الوضع القديم فقط) نسبة الـ validation من آخر النص |
| `dataset` | مجلد dataset جاهز من `data.prepare` (إذا وُجد يُتجاهل `data_dir`) |
| `sampling` | `weights` (حسب الحجم) أو `weights_language_balanced` (عربي/إنجليزي بالتساوي) |
| `context_length` | عدد الحروف التي يراها الموديل لكل توقع |
| `embedding_dim` | حجم الـ embedding لكل حرف |
| `hidden_size` | (MLP فقط) حجم الطبقة المخفية |
| `n_heads` | (Transformer فقط) عدد الـ attention heads |
| `n_layers` | (Transformer فقط) عدد الـ blocks |
| `dropout` | (Transformer فقط) نسبة الـ dropout |
| `batch_size` | عدد النوافذ في كل batch |
| `learning_rate` | معدل التعلم لـ AdamW |
| `max_steps` | عدد خطوات التدريب |
| `grad_clip` | أقصى norm للـ gradients (0 = بدون clipping) |
| `lr_schedule` | `constant` أو `cosine` |
| `warmup_steps` | زيادة خطية للـ learning rate من 0 في البداية |
| `min_lr` | نهاية الـ cosine schedule |
| `early_stopping_patience` | يتوقف التدريب بعد هذا العدد من التقييمات بدون تحسن في val (0 = معطل) |
| `eval_interval` | كل كم خطوة نقيّم ونحفظ checkpoint |
| `eval_batches` | عدد الـ batches في كل تقييم |

### لماذا هذا الحجم للـ Transformer؟
`n_layers=4, n_heads=4, embedding_dim=64, context_length=64` → **212,224 parameter**.
- الجهاز CPU فقط (4 cores): هذا الحجم يتدرب 2000 خطوة في حوالي دقيقتين، فنقدر نجرّب بسرعة.
- البيانات الحالية 2,088 حرف تدريب فقط؛ موديل أكبر كان سيحفظ النص أسرع، لا أكثر.
- `context_length=64` يغطي تقريبًا سطرًا كاملًا من النص (السطور 50–80 حرفًا).
- `head_dim = 64/4 = 16`.

## النتائج

### المرحلة الأولى (MLP)
loss من 4.2345 (≈ ln 69) إلى 0.2908 على train؛ val ينخفض إلى 2.32 عند الخطوة 200 ثم يرتفع (overfitting).

### المرحلة الثانية: MLP vs Transformer (نفس البيانات، نفس الـ seed، 2000 خطوة، CPU)

| | MLP | Transformer |
|---|---|---|
| parameters | 59,837 | 212,224 |
| context_length | 16 | 64 |
| زمن التدريب | 8.0 ث | 127.1 ث |
| initial loss (train / val) | 4.2345 / 4.2334 | 4.2601 / 4.2703 |
| best val loss | 2.3222 (step 200) | 2.3285 (step 200) |
| final val loss | 3.1975 | 3.7766 |
| final train loss | 0.2908 | 0.0924 |

**القراءة الصادقة لهذه الأرقام:**
- كلا الموديلين **يحفظ** نص التدريب (memorization)، ولا يوجد دليل على generalization: الـ val loss يبدأ بالارتفاع بعد الخطوة ~200 بينما الـ train loss يستمر بالنزول.
- الـ Transformer يحفظ أكثر (train 0.09) وبالتالي val أسوأ في النهاية (3.78). هذا ليس دليلًا أنه "أفضل" أو "أسوأ".
- أفضل val loss متساوٍ تقريبًا (2.32 مقابل 2.33). البيانات (233 حرف validation، وهي آخر 3 أسطر إنجليزية) لا تسمح بالحكم على أي architecture أفضل.
- الأرقام ليست قابلة للمقارنة تمامًا: context مختلف (16 مقابل 64)، dropout في الـ Transformer فقط، والتقييم كل 200 خطوة فقط.

## المرحلة الثالثة: corpus حقيقية وتقييم الـ generalization

```bash
.venv/bin/python -m data.download                       # المصادر العامة → data/raw/general/ + provenance.json
.venv/bin/python -m data.extract_repo --repo /path/to/after-seals \
    --include-pdf "assest/آلية عمل خدمة التوصيل في Odoo.pdf"   # domain corpus + held-out questions
.venv/bin/python -m data.prepare                        # → data/processed/hishamlm_v1/
.venv/bin/python -m data.stats --path data/processed/hishamlm_v1/train

.venv/bin/python -m training.train --config configs/transformer_corpus.json       # التدريب
.venv/bin/python -m training.evaluate --checkpoint checkpoints/transformer_corpus/best.pt     # loss + perplexity
.venv/bin/python -m inference.sample_eval --checkpoint checkpoints/transformer_corpus/best.pt # prompts + memorization
```

- `data/sources.py`: كل مصدر عام مع رابط مثبّت على commit محدد، الترخيص، و sha256. والمصادر المرفوضة مع السبب.
- `data/clean.py`: تنظيف محافظ (BOM، NFC، line endings، control chars، فراغات) بدون أي تغيير في الحروف العربية أو التشكيل.
- `data/extract_repo.py`: يستخرج نص domain من ريبو (قراءة فقط) مع استبعاد الأسرار والبيانات الشخصية والملفات المولّدة، ويعزل ملفات التقييم كـ held-out benchmark.
- `data/prepare.py`: dedup عبر كل المصادر، split بدون leakage (blocks للمصادر العامة، مستندات كاملة للـ domain)، vocabulary من train فقط، baselines، أوزان الـ sampling، و dataset fingerprint.

- `training/data.py`: كل مصدر stream مستقل؛ كل صف في الـ batch يختار مصدره حسب الأوزان (الـ weighting في الـ sampling، بدون نسخ ملفات).
- `training/evaluate.py`: كل حرف في val/test يُقيَّم مرة واحدة (sliding windows، كل توقع يرى ≥ 32 حرف سياق).
- `inference/sample_eval.py`: prompts ثابتة غير موجودة في train (`configs/eval_prompts.json`) + فحص memorization: أطول تطابق حرفي مع train ونسبة الـ 32-grams المنسوخة، مقارنةً بنصوص held-out حقيقية.
- كل run يكتب `run_meta.json`: seed، config، git commit، dataset fingerprint، vocab size، parameters، إصدارات PyTorch/Python، RAM.

مجموعات التقييم:
- **A** general validation/test — language modeling عام.
- **B** After-Sales held-out documents — نصوص domain لم تدخل التدريب.
- **C** After-Sales evaluation questions — محجوزة لتقييم instruction/chat لاحقًا، لا تدخل التدريب ولا الـ vocabulary.

### نتيجة المرحلة الثالثة (Transformer 237,696 parameter، 6000 خطوة، CPU، 405 ثانية)

dataset fingerprint `55c7694f…dbe1`، sampling متوازن لغويًا (عربي عام 41.25%، إنجليزي عام 41.25%، After-Sales 17.5%).

| المجموعة | loss | perplexity | unigram baseline | uniform baseline |
|---|---|---|---|---|
| train (sampling mix) | 1.961 | 7.11 | — | 5.591 |
| A general val | 1.929 | 6.88 | 3.652 | 5.591 |
| A general test | 1.945 | 6.99 | 3.665 | 5.591 |
| B After-Sales val | 1.850 | 6.36 | 3.934 | 5.591 |
| B After-Sales test | 2.017 | 7.52 | 4.004 | 5.591 |

- validation/test أقل بكثير من الـ unigram baseline، والفجوة بين train و val شبه معدومة: الموديل تعلّم أنماطًا تعمم على نص لم يره، ولم يحفظ النص.
- فحص الـ memorization: أطول تطابق حرفي بين المخرجات و train هو 12–17 حرفًا، أقل من نص held-out حقيقي (median 19.5)، و 0% من الـ 32-grams منسوخة.
- لكن المخرجات ما زالت في الغالب كلمات شبه صحيحة وليست جملًا ذات معنى: الموديل تعلّم إحصاءات الحروف والصرف وعلامات الترقيم وبعض الكلمات الشائعة. هو **underfitting** (val ما زال ينخفض عند الخطوة 6000).
- الـ prompts العربية العامة تنجرف نحو مفردات After-Sales (التوصيل، التركيب، الفاتورة) بسبب oversampling الـ domain.

## المرحلة الرابعة: BPE tokenizer مكتوب من الصفر + bits per character

```bash
.venv/bin/python -m tokenizer.train_bpe --vocab-size 4096          # merges من train فقط، بأوزان الـ sampling
.venv/bin/python -m training.train --config configs/phase4/bpe_4096.json
.venv/bin/python -m training.evaluate --checkpoint checkpoints/transformer_bpe4096/best.pt
.venv/bin/python -m inference.domain_drift --checkpoint checkpoints/transformer_bpe4096/best.pt --samples 50 --seed 5000
.venv/bin/python -m training.compare_runs transformer_corpus transformer_bpe1024 transformer_bpe2048 transformer_bpe4096
```

- `tokenizer/bpe_tokenizer.py`: pre-tokenization (كلمة مع المسافة قبلها، أرقام منفردة، رموز، مسافات) → حروف أساسية (تكرار ≥ 10 في train) → merges بالترتيب. الحروف النادرة/غير المعروفة تتحول إلى bytes الـ UTF-8 (byte fallback)، فلا يوجد `<unk>` و `decode(encode(text)) == text` دائمًا.
- **BPC** (bits per character) = loss بالـ bits ÷ عدد الحروف الأصلية التي تمثلها الـ tokens المتوقعة. هو المقياس الوحيد القابل للمقارنة بين tokenizers مختلفة.
- كل run يسجل `tokens_seen` و `chars_seen`، فنقارن عند نفس الحساب (نفس عدد الـ tokens) وعند نفس عدد الحروف المرئية.
- `lr_decay_steps`: الـ cosine ينتهي عند خطوة محددة ثم يثبت عند `min_lr`، فإكمال run لاحقًا لا يرفع الـ learning rate فجأة.

### النتائج (نفس البيانات والأوزان والـ seed والـ Transformer: 4 layers، 4 heads، dim 64، context 64 token)

| | Char | BPE 1K | BPE 2K | BPE 4K | BPE 4K، After-Sales 10% |
|---|---|---|---|---|---|
| parameters | 237,696 | 334,464 | 465,536 | 727,680 | 727,680 |
| tokens processed | 40.96M | 40.96M | 40.96M | 40.96M | 40.96M |
| characters represented | 41.0M | 82.1M | 99.7M | 116.7M | 114.8M |
| حروف لكل token / سياق بالحروف | 1.0 / 64 | 2.05 / 131 | 2.43 / 156 | 2.80 / 179 | 2.80 / 179 |
| زمن التدريب | 22.7 د | 25.6 د | 30.1 د | 38.5 د | 37.5 د |
| **A general test BPC** | 2.666 | 2.384 | 2.273 | **2.156** | 2.123 |
| B After-Sales test BPC | 2.703 | 2.402 | 2.327 | **2.214** | 2.276 |
| Arabic test BPC | 2.976 | 2.809 | 2.751 | **2.662** | 2.616 |
| English test BPC | 2.508 | 2.168 | 2.029 | **1.898** | 1.872 |
| val BPC عند 40M حرف مرئي | 2.685 | 2.458 | 2.308 | **2.179** | — |
| فجوة A val − train | −0.007 | 0.056 | 0.138 | 0.254 | 0.169 |
| أطول تطابق حرفي (max / median)، المرجع 31 / 19.5 | 18 / 14 | 26 / 16 | 51 / 16 | 41 / 20 | 31 / 18 |
| انجراف prompts العربية العامة نحو After-Sales (150 عينة) | **123/150** | 8/150 | 3/150 | 7/150 | 8/150 |

**القراءة:**
- BPE أفضل فعلًا وليس مجرد ضغط: BPC أقل عند نفس الحساب **وعند نفس عدد الحروف المرئية**، على كل المجموعات.
- التحسن في الإنجليزي (−24%) أكبر بكثير من العربي (−10.5%).
- جزء من التحسن قد يعود لزيادة الـ parameters (كلها في embedding و output layer)؛ التجربة لا تفصل الأثرين.
- مع vocab أكبر تكبر فجوة train/val، وتظهر جمل After-Sales محفوظة (33–51 حرفًا) لأن الـ domain يُرى ~100 مرة.
- الكلمات أصبحت حقيقية في الغالب، لكن الجمل ما زالت بلا معنى متماسك.
- الانجراف نحو After-Sales كان مشكلة character tokenizer أساسًا؛ تقليل حصة الـ domain إلى 10% لم يغير الانجراف (7 → 8 من 150)، لكنه حسّن General BPC (−0.03) وأضعف Domain BPC (+0.06) وقلل الـ overfitting.

## المرحلة الخامسة: أثر حجم الـ Transformer فقط

لم يتغير: BPE 4K، الـ corpus والـ splits (fingerprint `55c7694f…`)، أوزان الـ sampling (41.25/41.25/17.5)، context 64 token، الـ seed، الـ prompts، فحوص الـ memorization، جدول الـ LR (warmup 200، cosine حتى 6000 ثم ثابت 1e-4)، LR = 1e-3 لكل الأحجام (كلها مستقرة في الـ smoke test).
تغيّر فقط: `embedding_dim` و `n_layers` و `n_heads` (head size ثابت 32 للأحجام الجديدة)، و `tie_weights` في تجربة منفصلة.

```bash
.venv/bin/python -m training.train --config configs/phase5/bpe4k_1.3m_tied.json --max-steps 20000
.venv/bin/python -m training.compare_runs transformer_bpe4096 transformer_bpe4k_1.3m_tied_20k transformer_bpe4k_3.5m_tied_20k transformer_bpe4k_7.4m_tied_6k
```

**Weight tying** (`tie_weights: true`): طبقة الإخراج تستخدم نفس مصفوفة الـ token embedding. عند كل حجم كانت النسخة المربوطة أصغر وبنفس السرعة وأفضل على كل مجموعات التقييم، مع فجوة train/val أصغر.

| | 728K (baseline) | 1.3M tied | 1.3M tied | 3.5M tied | 7.4M tied | 8.4M |
|---|---|---|---|---|---|---|
| C × L × H | 64×4×4 | 128×4×4 | 128×4×4 | 192×6×6 | 256×8×8 | 256×8×8 |
| steps (best / done) | 19,750 / 20,000 | 6,000 | 20,000 | 7,750 / 9,250 (early stop) | 6,000 | 5,750 / 6,000 |
| characters represented | 116.7M | 35.0M | 116.7M | 54.0M | 35.0M | 35.0M |
| زمن التدريب / RAM | 38.5 د / 0.89 GB | 16.7 د / 0.96 GB | 54.4 د / 0.97 GB | 49.6 د / 1.20 GB | 57.8 د / 1.51 GB | 58.1 د / 1.54 GB |
| train BPC | 1.893 | 1.786 | 1.623 | 1.489 | 1.315 | 1.308 |
| A general val / test | 2.147 / 2.156 | 2.049 / 2.052 | 1.988 / **1.985** | 1.952 / 1.947 | 1.912 / **1.902** | 1.961 / 1.956 |
| B After-Sales val / test | 2.036 / 2.214 | 1.945 / **2.117** | 1.967 / 2.156 | 1.994 / 2.209 | 2.013 / 2.201 | 2.050 / 2.250 |
| Arabic / English test | 2.662 / 1.898 | 2.486 / 1.830 | 2.419 / 1.765 | 2.392 / 1.722 | 2.351 / 1.674 | 2.459 / 1.700 |
| فجوة val − train | 0.254 | 0.263 | 0.365 | 0.464 | 0.597 | 0.653 |
| أطول تطابق حرفي (max / median)، المرجع 31 / 19.5 | 41 / 20 | 39 / 19 | **32 / 19** | 72 / 20 | 79 / 25 | 38 / 24 |

**القراءة:**
- النص العام ما زال **capacity-limited**: كل تكبير يحسّن General BPC عند نفس البيانات، لكن التحسن يتناقص.
- After-Sales **data-limited**: النماذج الأكبر والتدريب الأطول يحسّنان train ويسوءان domain test، والحفظ الحرفي للـ domain يزيد مع الحجم.
- 3.5M tied توقف مبكرًا عند ~45M حرف: أول علامة واضحة على حد البيانات.
- عند نفس الحساب (0.10 PFLOPs) الأفضل 1.3M tied (val 2.050) مقابل 2.147 للـ baseline و 2.148 لـ 3.5M tied.
- **الإعداد المختار: 1.3M tied، 20K خطوة** — أفضل توازن بين BPC والحفظ والوقت والذاكرة.

## Dataset v2: كتب هنداوي العربية (خاصة)

v2 = v1 + أربعة كتب عربية من هنداوي (1.38M حرف بعد التنظيف)، مستخدمة بإذن استخدام شخصي موقّع — **لا تُعاد مشاركتها ولا يُنشر موديل مدرّب عليها**. الكتب خارج git (`data/raw/private_books/`) ولا تدخل إلا بـ `--include-private`. مجموعات A و B لم تتغير؛ للكتب مجموعة تقييم خاصة `A_books`.

```bash
.venv/bin/python -m data.download --include-private
.venv/bin/python -m data.prepare --include-private --out-dir data/processed/hishamlm_v2
.venv/bin/python -m training.train --config configs/v2/bpe4k_1.3m_tied_v2.json
.venv/bin/python -m training.evaluate --checkpoint checkpoints/transformer_bpe4k_1.3m_tied_20k/best.pt --dataset data/processed/hishamlm_v2 --tag v2sets
.venv/bin/python -m inference.sample_eval --checkpoint ... --prompts configs/eval_prompts_long_ar.json --max-new-chars 400 --repeats 3 --tag long_ar
.venv/bin/python -m inference.arabic_quality --samples .../samples_best_long_ar.json ...
```

### v1 مقابل v2 (نفس الموديل 1.3M tied، نفس BPE 4K، نفس الـ seed، 20K خطوة = 40.96M token، من random init)

| test BPC | v1 | v2 | الفرق |
|---|---|---|---|
| Books | 3.020 | **2.286** | −0.733 |
| كل العربي (العربي القديم + الكتب) | 2.764 | **2.376** | −0.388 |
| العربي القديم (CV + NTREX + XQuAD) | **2.419** | 2.498 | +0.079 |
| English | 1.765 | **1.758** | −0.007 |
| General A | **1.985** | 2.007 | +0.022 |
| After-Sales B | 2.156 | **2.090** | −0.067 |

- العربي القديم ساء لأن الكتب أخذت 58% من حصته في الـ sampling (CV العربي من 24.9% إلى 10.5%) وأسلوبه مختلف (جمل قصيرة وأخبار).
- الكلمات العربية غير الموجودة في معجم التدريب (prompts عربية طويلة، 18 عينة × 400 حرف): 20.5% → **11.5%** بمعجم v2، و 21.7% → 19.3% بمعجم v1 (بدون كتب)؛ النص الحقيقي 9.4% / 14.1%.
- الجمل أطول (11.7 → 19.6 كلمة) وبروابط فقرات حقيقية، لكن المعنى ما زال غير مترابط، وظهر انجراف نحو مواضيع الكتب (الحوسبة، البيانات).
- لا نسخ من الكتب: أطول تطابق في الـ prompts العربية الطويلة 29 حرفًا (المرجع 31).
- BPE 4K القديم يغطي الكتب بكفاءة (2.53 حرف/token)؛ الـ byte fallback 0.25% من الحروف، كله تقريبًا أرقام عربية مشرقية (٠-٩).

### v2: BPE القديم (من v1) مقابل BPE جديد (من v2) — نفس الموديل والبيانات والـ seed

```bash
.venv/bin/python -m tokenizer.train_bpe --dataset data/processed/hishamlm_v2 --vocab-size 4096
.venv/bin/python -m tokenizer.coverage --tokenizer data/processed/hishamlm_v2/tokenizers/bpe_4096.json --dataset data/processed/hishamlm_v2
.venv/bin/python -m training.train --config configs/v2/bpe4k_v2tok_1.3m_tied_v2.json
```

| | BPE القديم | BPE الجديد | الفرق |
|---|---|---|---|
| حروف/token: الكتب / العربي القديم | 2.53 / 2.43 | **2.79** / 2.36 | +10% / −3% |
| byte fallback في الكتب | 0.25% من الحروف | **0.007%** | الأرقام العربية المشرقية صارت حروفًا أساسية |
| Books test BPC | 2.286 | **2.246** | −0.040 |
| كل العربي test BPC | 2.376 | **2.363** | −0.013 |
| العربي القديم test BPC | **2.498** | 2.521 | +0.023 |
| English test BPC | 1.758 | 1.755 | −0.003 |
| General A test BPC | **2.007** | 2.013 | +0.006 |
| After-Sales test / val BPC | **2.090** / **1.873** | 2.147 / 1.884 | +0.057 / +0.011 |
| A val / B val عند نفس الحروف (116M) | 2.030 / 1.898 | 2.034 / 1.895 | ≈ 0 |
| كلمات غير معروفة (معجم v2 / v1) | 11.5% / 19.3% | 11.8% / 17.2% | مختلط |

الخلاصة: أثر الـ tokenizer الجديد صغير ويعيد التوزيع (أفضل للكتب، أسوأ قليلًا للعربي القصير و After-Sales)؛ التحسن الحقيقي جاء من البيانات (الكتب) وليس من الـ tokenizer. تجربة واحدة بـ seed واحد، والفروق ±0.01–0.02 قريبة من مستوى الضجيج.

## المرحلة السادسة: context 64 مقابل 128 (الـ stack الرسمي، متغير واحد فقط)

الـ stack: dataset v2 (مع الكتب الخاصة)، BPE 4K القديم من v1، 1.3M tied (128 × 4 layers × 4 heads)، نفس الـ sampling والـ seed والـ optimizer والـ LR schedule (warmup 200، cosine إلى 6000، ثم 1e-4).
الفرق الوحيد بين `configs/v2/bpe4k_1.3m_tied_v2.json` و `configs/phase6/bpe4k_1.3m_tied_v2_ctx128.json` هو `context_length` (و `run_name`).

```bash
.venv/bin/python -m tokenizer.context_coverage --tokenizer data/processed/hishamlm_v1/tokenizers/bpe_4096.json --dataset data/processed/hishamlm_v2
.venv/bin/python -m training.train --config configs/phase6/bpe4k_1.3m_tied_v2_ctx128.json --max-steps 10000   # = حروف ctx64@20K
.venv/bin/python -m training.train --config configs/phase6/bpe4k_1.3m_tied_v2_ctx128.json --resume --max-steps 20000
.venv/bin/python -m training.evaluate --checkpoint checkpoints/transformer_bpe4k_1.3m_tied_v2_ctx128/best.pt --context-length 64 --tag ctx64window
.venv/bin/python -m inference.coherence --checkpoints ctx64=checkpoints/transformer_bpe4k_1.3m_tied_v2_20k/best.pt \
    ctx128=checkpoints/transformer_bpe4k_1.3m_tied_v2_ctx128/best.pt --samples-per-prompt 5 --out coherence.json
```

**تصميم المقارنة العادلة.** بنفس batch 32 يعالج ctx128 ضعف الـ tokens في كل خطوة، لذلك لا نقارن بعدد الخطوات:
- نقطة 1 (نفس الحروف ونفس الـ compute تقريبًا): ctx128 عند 10K خطوة مقابل ctx64 الرسمي عند 20K (كلاهما 117.0M حرف، 40.96M token).
- نقطة 2 (ميزانية أكبر): ctx128 عند 20K (234M حرف) مقابل نسخة من ctx64 أكملت التدريب حتى 40K (توقف early stopping عند 29,750، أفضل checkpoint عند 28,250 = 165M حرف). الـ checkpoint الرسمي لم يُلمس.

**كم نصًا يغطي الـ context** (نوافذ متتالية من train split، BPE القديم):

| المجموعة | حروف/token | 64 token: حروف / كلمات / نهايات جمل | 128 token: حروف / كلمات / نهايات جمل | جمل تتسع كاملة في 64 / 128 |
|---|---|---|---|---|
| عربي عام (CV، NTREX، XQuAD) | 2.50 | 160 / 27.6 / 6.8 | 319 / 54.8 / 13.6 | 100% / 100% (median الجملة 8 tokens) |
| English | 2.92 | 187 / 35 / 5.2 | 374 / 70 / 10.4 | 100% / 100% (median 15) |
| After-Sales | 3.35 | 214 / 33.6 / 5.2 | 429 / 67.0 / 10.4 | 98.4% / 99.9% (median 13) |
| الكتب الخاصة | 2.59 | 166 / 27.8 / 1.1 | 331 / 55.2 / 2.2 | **55.5% / 90.3%** (median 60) |

الكتب هي المجموعة الوحيدة التي لا تتسع جملها في 64 token؛ هنا كان يُتوقع أكبر أثر.
(تصحيح لاحق: عدّ الكلمات كان يقسم الكلمة عند التشكيل؛ الأرقام أعلاه بعد الإصلاح، والفرق ≤ 1.5 كلمة.)

**التكلفة الحقيقية على هذا الـ CPU** (4 أنوية، بدون GPU):

| | ctx64 | ctx128 |
|---|---|---|
| parameters | 1,324,288 | 1,332,480 (+8,192 position embeddings) |
| زمن الخطوة (micro-benchmark، بدون eval) | 155–182 ms | 385 ms (×2.1–2.5) |
| tokens/sec / حروف/sec (benchmark) | 11.2–13.2K / 32–38K | 10.6K / 30.4K (−10 إلى −20%) |
| s/step مع الـ evals (تشغيل نظيف) | 0.199 | 0.425–0.430 |
| زمن التدريب: نقطة 1 / نقطة 2 | 76 min* / 103.5 min حتى أفضل خطوة | 72 min / 142.5 min |
| peak RAM | 982–988 MB | 1,345–1,350 MB (+37%) |
| FLOPs تقديرية لكل token | 8.28M | 8.67M (+4.7%) |

\* الـ 76 دقيقة لـ ctx64 كانت مع تنافس على الـ CPU؛ بسرعة نظيفة ≈ 66 دقيقة.

**BPC (تقييم كامل بنوافذ متداخلة، stride = نصف الـ context):**

| المجموعة (test) | ctx64@20K | ctx128@10K (نفس الحروف) | ctx64@28.25K (165M حرف) | ctx128@20K (234M حرف) |
|---|---|---|---|---|
| General A (val / test) | **2.003 / 2.007** | 2.043 / 2.051 | **1.980 / 1.983** | 2.013 / 2.019 |
| Books (val / test) | **2.303 / 2.286** | 2.338 / 2.327 | **2.275 / 2.262** | 2.298 / 2.289 |
| العربي القديم | **2.498** | 2.583 | **2.467** | 2.539 |
| كل العربي | **2.376** | 2.436 | **2.349** | 2.396 |
| English | **1.758** | 1.781 | **1.737** | 1.755 |
| After-Sales (val / test) | **1.873 / 2.090** | 1.915 / 2.173 | 1.891 / 2.118 | 1.961 / 2.229 |
| train (mix) / gap A val − train | 1.718 / 0.285 | 1.733 / 0.310 | 1.669 / 0.310 | 1.656 / 0.356 |

- عند نفس الحروف و نفس الـ compute تقريبًا، ctx64 أفضل في **كل** المجموعات (+0.02 إلى +0.09 BPC لصالحه).
- عند كل نقطة مشتركة في منحنى التدريب (15M → 174M حرف، 30 → 108 دقيقة) يبقى ctx64 متقدمًا بـ 0.02–0.04 val BPC؛ ctx128 لا يلحق.
- ctx128@20K بضعف الحروف وضعف الوقت ما زال أسوأ من ctx64@28K، و After-Sales يسوء معه (overfitting للـ domain الصغير: 57 مرورًا على النص).
- **هل يستعمل الموديل السياق الإضافي؟** تقييم ctx128 بنوافذ 64 فقط يعطي تقريبًا نفس الأرقام: General 2.013 مقابل 2.013، Books val 2.303 مقابل 2.298، After-Sales val 1.974 مقابل 1.961. أي أن الـ tokens من 65 إلى 128 تضيف ≤ 0.005–0.013 BPC: الموديل الصغير بالكاد يستفيد منها.

**مقاييس ترابط تلقائية** (`inference/coherence.py`، 15 prompt أطول من 64 token تذكر الموضوع في بدايتها: 6 عربي، 3 English، 3 After-Sales، 3 بأسلوب الكتب؛ 5 عينات لكل prompt بنفس الـ seeds، 400 حرف، temperature 0.8، top-k 20). هذه مقاييس سطحية ولا تقيس الفهم، والعينات قليلة (15–30 لكل مجموعة):

| | ctx64@20K | ctx64@28K | ctx128@10K | ctx128@20K | نص حقيقي |
|---|---|---|---|---|---|
| عربي: كلمات مخترعة | 13.9% | 14.5% | 16.1% | **12.5%** | 10.8% |
| عربي: كلمات الموضوع /100 كلمة | 0.20 | 0.20 | 0.14 | 0.24 | – |
| عربي: انتقالات مفاجئة | 0.55 | 0.56 | 0.50 | **0.40** | 0.39 |
| English: كلمات الموضوع /100 كلمة | 0.16 | 0.08 | 0.63 | 0.07 | – |
| After-Sales: كلمات الموضوع / ما زال في الثلث الأخير | 9.3 / 53% | 9.2 / 67% | 11.1 / **80%** | 11.2 / **80%** | – |
| After-Sales: fragments / templates | 9% / 1% | 11% / 14% | 16% / 5% | 5% / 2% | 23% / 1% |
| الكتب: كلمات الموضوع / الثلث الأخير | 3.7 / 40% | 5.2 / 60% | 6.9 / 53% | **10.1 / 60%** | – |
| الكتب: كلمات مخترعة | 12.3% | 11.9% | 13.0% | 10.7% | 9.2% |
| تكرار 4-grams (كل المجموعات) | ≤ 0.8% | ≤ 0.3% | ≤ 0.7% | ≤ 0.5% | – |

- على العربي العام و English: كل الموديلات تفقد الموضوع تقريبًا فورًا (كلمات الموضوع ≈ 0) وتنجرف إلى أسلوب الكتب (القرن التاسع عشر، الآلة، البيانات) أو إلى حوارات Gutenberg. السياق الأطول لم يغيّر ذلك.
- على الكتب و After-Sales: ctx128 يُبقي كلمات الموضوع أكثر (الكتب ≈ ×2)، لكن جزءًا من ذلك تكرار لكلمة الموضوع نفسها ("البيانات البيانات")، و distinct-2 أقل قليلًا (0.96 مقابل 0.98).
- الكلمات المخترعة تتبع الـ BPC أكثر من الـ context: ctx128@10K أسوأ (16.1%)، و ctx128@20K أفضل لأنه تدرّب على ضعف الحروف.

**Memorization** (مقابل train كاملًا؛ النص الحقيقي غير المرئي: median 21.5، max 31):
- prompts القديمة: max / median أطول تطابق حرفي: ctx64@20K 49 / 19، ctx64@28K **79 / 19 (تحذير نسخ واحد: سطر After-Sales من 79 حرفًا)**، ctx128@10K 35 / 20، ctx128@20K 38 / 24.
- prompts الترابط: نسبة 50-char overlap ≤ 0.4% لكل الموديلات؛ أطول تطابق ≥ 45 حرفًا يظهر فقط في After-Sales (ملفات `000_MEASUREMENT_SERVICE_FLOW_AUDIT`، `044_pageKnowledge`، `046_workflow-responsibility-audit`، `002_app`). الكتب والعربي العام: ≤ 31 حرفًا، أي ضمن مستوى النص الحقيقي.
- مصدر أطول تطابق في prompts العربي العام هو الكتب في 17–24 من 30 عينة: أسلوب الكتب يهيمن على العربي المولَّد.
- السياق الأطول لم يزد النسخ الحرفي؛ ما زاده هو كثرة المرور على بيانات After-Sales الصغيرة.

**القرار:** لا يُعتمد ctx128 على هذا الحجم وهذا الـ CPU. عند نفس الحروف ونفس الـ compute هو أسوأ في الـ BPC في كل المجموعات، والموديل لا يستعمل السياق الإضافي فعليًا (تقييم النوافذ 64)، ويكلف ×2.1 زمن للخطوة و +37% RAM. الفائدة الوحيدة المتسقة: إبقاء كلمات الموضوع في الكتب و After-Sales، وهي فائدة سطحية جزئيًا. الـ checkpoint الموصى به يبقى `transformer_bpe4k_1.3m_tied_v2_20k` (ctx64). نسخة ctx64@28K أفضل في الـ BPC العام لكنها أسوأ في After-Sales وأظهرت تحذير نسخ، لذلك لا تحل محل الرسمي دون قرار. تجربة واحدة بـ seed واحد.

## دفعة كتب عربية ثانية: فحص فقط، بدون تدريب

```bash
.venv/bin/python -m data.books_candidate --raw data/raw/private_books/books_ar_batch2 \
    --books-config configs/private_books_batch2.json --checkpoint checkpoints/transformer_bpe4k_1.3m_tied_v2_20k/best.pt
```

11 ملف EPUB جديد (commit `06a24b0`، Git LFS، 6.56 MB)، نُزّلت إلى `data/raw/private_books/books_ar_batch2/` وتطابقت كلها مع hash الـ LFS. النصوص المستخرجة في `data/processed/` (خارج git). الـ tokenizer والـ dataset الرسمية والـ fingerprint لم تتغير.

ما يُحذف (بدون إعادة صياغة): الغلاف، صفحات العنوان والملاحظات و about، الفهارس التي هي قوائم فقط، الحواشي الختامية، رؤوس الصفحات وأرقامها، العناوين المتكررة، الروابط وحسابات التواصل، علامات الويكي، أرقام الإحالة إلى الحواشي. التغييرات الحرفية الوحيدة (مسجلة لكل كتاب): presentation forms → حروف عادية، وأشكال الحروف الفارسية التي يضعها الـ OCR (ک ڪ ی ھ ۰-۹) → حروف وأرقام عربية.

درجة الاستخراج: A نظيف؛ B نظيف الحروف لكن بلا ترقيم؛ C ضجيج OCR كبير (≥ 25 علامة خطأ OCR لكل 1000 كلمة: «ء» بدل الفاصلة، «»» بلا «« »، حروف فارسية، رموز لاتينية قصيرة)؛ F لا شيء صالح.

| الكتاب | النوع | المستخرج | الدرجة | حروف/token | BPC بالموديل الرسمي | في الـ candidate |
|---|---|---|---|---|---|---|
| الجريمة والعقاب (2) — دوستويفسكي، ترجمة سامي الدروبي | EPUB ناشر | 525K | A | 2.42 | 3.24 | نعم (الحقوق غير مؤكدة) |
| كليلة ودمنة — ابن المقفع | Wikisource | 241K | A | 2.38 | 3.39 | نعم |
| النظرات (البخيل) — المنفلوطي | Wikisource | 7K | A | 2.21 | 3.31 | نعم |
| البيان والتبيين — الجاحظ | Wikisource | 806K | B (لا ترقيم، همزات محذوفة) | 2.41 | 3.75 | نعم (كلاسيكي، بوزن منخفض) |
| بداية المجتهد 1 و 2 — ابن رشد | OCR (Internet Archive) | 608K + 637K | C | 1.96 / 1.99 | 4.23 / 4.19 | لا |
| الخيميائي — كويلو، ترجمة جواد صيداوي | OCR | 173K | C | 2.40 | 3.49 | لا |
| دماغك تحت تأثير الإباحية — غاري ويلسون | OCR | 350K | C | 2.45 | 3.45 | لا |
| adab | OCR | 0 (نص لاتيني عشوائي) | F | – | – | لا |
| العقد الفريد | Wikisource | 4K (فهرس روابط فقط) | F | – | – | لا |
| مجلة الرسالة | Wikisource | 0 (1,018 صفحة فهرس عناوين) | F | – | – | لا |

للمقارنة: الموديل الرسمي يعطي 2.29 BPC على كتب هنداوي (test) و 2.50 على العربي القديم.

- لا تكرار: 0 فقرات near-duplicate، تكرار حرفي ≤ 2 فقرة لكل كتاب، و 0% من نوافذ 50 حرفًا موجودة في v2 (العربي العام، كتب هنداوي، After-Sales).
- الـ candidate (`data/processed/private_books_v2_candidate/`): 1.58M حرف (تراث 1.05M، رواية 0.53M، مقالات 7K). هذا +66% على العربي الحالي (2.38M) لو دخل كله.
- الأرقام العربية المشرقية (٠-٩) ليست في حروف الـ BPE القديم؛ لذلك 2.7% من حروف «بداية المجتهد» تذهب إلى byte fallback.

## ما الذي ما زال غير موجود
- بيانات تدريب حقيقية بحجم معقول (الشرط الأساسي لأي مقارنة ذات معنى).
- بيانات أكثر (خصوصًا After-Sales وعربي أصلي) — هي الحد الحالي، لا حجم الموديل.
- مقاييس تقييم غير الـ loss.
- لا يوجد RAG، chat، API، أو واجهة — عن قصد.
