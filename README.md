# HishamLM

نموذج لغوي (Language Model) نبنيه من الصفر: أوزان تبدأ عشوائيًا، architecture مكتوبة يدويًا،
بدون أي pretrained model أو مكتبة tokenizer أو Hugging Face. PyTorch مستخدم فقط كإطار
tensors / autograd / GPU.

**الحالة الحالية: المرحلة الأولى** — إثبات أن الـ pipeline كاملًا يعمل (tokenizer → data → model → training → checkpoint → generation).

## بنية المشروع

```
HishamLM/
├── configs/tiny_char.json        كل إعدادات التدريب في ملف واحد
├── data/
│   ├── raw/                      ملفات .txt للتدريب (فيها sample صغير للاختبار)
│   └── processed/                محجوز للمراحل القادمة
├── tokenizer/char_tokenizer.py   Character-level tokenizer
├── model/mlp_lm.py               MLP Language Model
├── training/
│   ├── config.py                 TrainConfig (dataclass يُقرأ من JSON)
│   ├── data.py                   قراءة النصوص، split، batching
│   ├── utils.py                  seed و device
│   └── train.py                  training loop
├── inference/generate.py         توليد نص token-by-token
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

### 3. Model (`model/mlp_lm.py`)
MLP Language Model (على فكرة Bengio 2003)، وهو أبسط نموذج عصبي منطقي يستخدم سياقًا أطول من حرف واحد:

```
token ids (B, T)
  → Embedding                      (B, T, C)
  → لكل موضع t: ضم embeddings آخر n حروف (t-n+1 … t)   (B, T, n·C)
  → Linear + tanh                  (B, T, hidden)
  → Linear                         (B, T, vocab)   = logits للحرف التالي
```

- `n = context_length`، والمواضع الأولى تُكمَّل بـ zero vectors من اليسار.
- النموذج causal: كل موضع يرى فقط الحروف السابقة له (مُختبر في `tests/test_model.py`).
- طبقة الإخراج تُصغَّر عند الـ initialization حتى يبدأ الـ loss قريبًا من `ln(vocab_size)` (تخمين عشوائي منتظم).
- الـ interface هو `(B, T) → (B, T, vocab)`، نفس ما سيستخدمه Transformer لاحقًا، فيبقى الـ data pipeline والـ training loop كما هما.

### 4. Training loop (`training/train.py`)
لكل خطوة: batch → forward → cross-entropy loss → `backward()` → `optimizer.step()` (AdamW) → `zero_grad()`.

- قبل التدريب (step 0) وكل `eval_interval` خطوة: حساب train/val loss على نفس الـ batches الثابتة كل مرة (حتى تكون الأرقام قابلة للمقارنة).
- كل تقييم يحفظ checkpoint في `checkpoints/<run_name>/ckpt.pt` يحتوي: أوزان النموذج، حالة الـ optimizer، الـ config، الـ vocabulary، رقم الخطوة، والـ losses.
- أيضًا يُكتب `log.csv` (منحنى الـ loss) و`vocab.json`.
- seed ثابت (`seed` في الـ config) → نفس النتائج تمامًا عند إعادة التشغيل.

### 5. Generation (`inference/generate.py`)
يحمّل الـ checkpoint، يحوّل الـ prompt إلى IDs، ثم في حلقة: يمرر آخر `context_length` token للنموذج، يأخذ logits آخر موضع، يقسم على `temperature`، (اختياريًا `top_k`)، softmax، ثم يسحب token واحد ويضيفه للسلسلة.

## التشغيل

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# الاختبارات
.venv/bin/python -m unittest discover tests -v

# التدريب
.venv/bin/python -m training.train --config configs/tiny_char.json

# التوليد
.venv/bin/python -m inference.generate --prompt "The cat" --max-new-tokens 200
.venv/bin/python -m inference.generate --prompt "القط" --temperature 1.0 --top-k 10 --seed 7
```

كل الأوامر تُشغَّل من جذر المشروع. لإضافة بيانات: ضع ملفات `.txt` (UTF-8) في `data/raw/`.

## الإعدادات (`configs/tiny_char.json`)

| المفتاح | المعنى |
|---|---|
| `seed` | الـ random seed |
| `device` | `auto` أو `cpu` أو `cuda` |
| `val_fraction` | نسبة الـ validation من آخر النص |
| `context_length` | عدد الحروف التي يراها النموذج لكل توقع |
| `embedding_size` | حجم الـ embedding لكل حرف |
| `hidden_size` | حجم الطبقة المخفية |
| `batch_size` | عدد النوافذ في كل batch |
| `learning_rate` | معدل التعلم لـ AdamW |
| `max_steps` | عدد خطوات التدريب |
| `eval_interval` | كل كم خطوة نقيّم ونحفظ checkpoint |
| `eval_batches` | عدد الـ batches في كل تقييم |

## نتيجة أول تشغيل (CPU)

- vocab: 69 حرفًا (عربي + إنجليزي + `<unk>`)، 2,321 حرفًا للتدريب، 59,837 parameter.
- loss: من **4.2345** (≈ ln 69 = 4.2341) إلى **0.2908** على train خلال 2000 خطوة (~7 ثوانٍ).
- val loss ينخفض إلى 2.32 عند الخطوة 200 ثم يرتفع: **overfitting** متوقع لأن البيانات صغيرة جدًا (النموذج يحفظ النص). هذا مقصود في هذه المرحلة.

## ما الذي ما زال غير موجود
- بيانات تدريب حقيقية بحجم معقول.
- Tokenizer أفضل (BPE مثلًا) — الحالي حرفي فقط.
- Transformer (self-attention، positional embeddings، layer norm…).
- learning rate schedule، gradient clipping، استئناف التدريب من checkpoint، حفظ أفضل checkpoint حسب val loss.
- مقاييس تقييم غير الـ loss (perplexity، إلخ).
- لا يوجد RAG، chat، API، أو واجهة — عن قصد.
