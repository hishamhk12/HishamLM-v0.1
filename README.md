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

## ما الذي ما زال غير موجود
- بيانات تدريب حقيقية بحجم معقول (الشرط الأساسي لأي مقارنة ذات معنى).
- Tokenizer أفضل (BPE مثلًا) — الحالي حرفي فقط.
- weight tying، وأي تكبير للموديل.
- مقاييس تقييم غير الـ loss.
- لا يوجد RAG، chat، API، أو واجهة — عن قصد.
