"""Registry of the general (non-domain) text sources.

Every source is pinned to an exact commit and file hash, and carries its license.
A source whose license is unclear does not go in this list (see REJECTED).
"""

GITHUB_RAW = "https://raw.githubusercontent.com"

SOURCES = [
    {
        "name": "common_voice_ar",
        "language": "ar",
        "description": "Mozilla Common Voice sentence collector, Arabic. Filtered to modern prose sentences.",
        "url": f"{GITHUB_RAW}/common-voice/common-voice/7a88506c8a0ffbffb0ba5e8117d7975c5fef7c36/server/data/ar/sentence-collector.txt",
        "repository": "https://github.com/common-voice/common-voice (server/data/ar)",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_note": "Sentences are submitted under CC0. Many look like translated Tatoeba sentences "
        "(Tatoeba: CC BY 2.0 FR); both licenses allow this use, attribution is kept here.",
        "sha256": "438817d8a0de050edfe4d1da10c7b3d9d77702db44d446f9e08e53f012f6e27e",
    },
    {
        "name": "ntrex_ar",
        "language": "ar",
        "description": "NTREX-128: 1,997 news sentences professionally translated to Modern Standard Arabic.",
        "url": f"{GITHUB_RAW}/MicrosoftTranslator/NTREX/468c6b69c7f6a75d31d4743d9daba2af566cc18d/NTREX-128/newstest2019-ref.arb.txt",
        "repository": "https://github.com/MicrosoftTranslator/NTREX",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_note": "Federmann et al., NTREX-128, SUMEval 2022.",
        "sha256": "d271c8645933421f760eba8ed8cc1ceaaa052f917d9c2839b8abb27e0ded71e6",
    },
    {
        "name": "xquad_ar",
        "language": "ar",
        "description": "XQuAD: 240 Wikipedia paragraphs (48 articles) professionally translated to Arabic. Paragraphs only.",
        "url": f"{GITHUB_RAW}/google-deepmind/xquad/7d30520c717524000f0d9d2f9c10a069acd9d285/xquad.ar.json",
        "repository": "https://github.com/google-deepmind/xquad",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_note": "Artetxe et al., On the Cross-lingual Transferability of Monolingual Representations, 2020.",
        "sha256": "abdabd7afed5c635d99cca0f3f0d0c9d9ed0bc77451e963c2e4e0638c29e486d",
    },
    {
        "name": "common_voice_en",
        "language": "en",
        "description": "Mozilla Common Voice sentence collector, English. A seeded random subset is used.",
        "url": f"{GITHUB_RAW}/common-voice/common-voice/7a88506c8a0ffbffb0ba5e8117d7975c5fef7c36/server/data/en/sentence-collector.txt",
        "repository": "https://github.com/common-voice/common-voice (server/data/en, LICENSE: CC0 1.0)",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_note": "",
        "sha256": "31ac8e200449ebd7aee4a8c6bd16d4dafd076cf11f7c3efe1035f0283712d136",
        "max_chars": 800_000,
    },
    {
        "name": "gutenberg_en",
        "language": "en",
        "description": "Project Gutenberg books from the NLTK gutenberg corpus (prose, 1865-1918).",
        "url": f"{GITHUB_RAW}/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/corpora/gutenberg.zip",
        "repository": "https://github.com/nltk/nltk_data (packages/corpora/gutenberg.zip)",
        "license": "Public domain (US)",
        "license_url": "https://www.gutenberg.org/policy/permission.html",
        "license_note": "Texts are public domain in the US; Project Gutenberg headers were already removed by NLTK.",
        "sha256": "2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611",
        "books": [
            "carroll-alice.txt",
            "chesterton-brown.txt",
            "chesterton-thursday.txt",
            "bryant-stories.txt",
            "burgess-busterbrown.txt",
        ],
    },
]

# Sources that may NOT be redistributed. They are used only when explicitly requested
# (python -m data.download --include-private / python -m data.prepare --include-private),
# stored under data/raw/private_books/ (ignored by git), and any model trained on them stays private.
_BOOKS_COMMIT = "e586234dfd07258590ba8cc3ce7b0e1dd40f03f6"
_BOOKS_URL = f"https://media.githubusercontent.com/media/hishamhk12/HishamLM-v0.1/{_BOOKS_COMMIT}/data/private_books/raw/ar"

PRIVATE_SOURCES = [
    {
        "name": "hindawi_books_ar",
        "language": "ar",
        "eval_group": "books",  # evaluated separately, so the v1 sets A and B keep their meaning
        "private": True,
        "description": "Four Arabic books published by Hindawi (EPUB), uploaded by the project owner.",
        "repository": f"https://github.com/hishamhk12/HishamLM-v0.1 (commit {_BOOKS_COMMIT}, "
                      "data/private_books/raw/ar, Git LFS)",
        "license": "All rights reserved (per each book's copyright page); used under the project owner's "
                   "signed personal-use permission",
        "license_url": "",
        "license_note": "Do not redistribute these books or publish models trained on them.",
        "files": [
            {"file": "the_machine.epub", "title": "الآلة قوة وسلطة", "author": "آر إيه بوكانان",
             "rights": "Hindawi (design); translation text: شوقي جلال",
             "url": f"{_BOOKS_URL}/13637483.epub",
             "sha256": "3b699a5d4efe8bde10c21fbf47a624a3ffb736db128c000fef35b0a91429c682"},
            {"file": "computational_thinking.epub", "title": "التفكير الحوسبي", "author": "بيتر جيه دنينج وماتي تيدري",
             "rights": "Hindawi (design, Arabic translation); original text © 2019 MIT",
             "url": f"{_BOOKS_URL}/%D8%A7%D9%84%D8%AA%D9%81%D9%83%D9%8A%D8%B1_%D8%A7%D9%84%D8%AD%D9%88%D8%B3%D8%A8%D9%8A.epub",
             "sha256": "b6598e263eb618ac9a89cf1b8f5deaf1607a99926ee7039dabe7b0b8031c56a2"},
            {"file": "theatre_digital_world.epub", "title": "المسرح والعالم الرقمي", "author": "أنطونيو بيتزو",
             "rights": "Hindawi (design); translation text: د. أماني فوزي حبشي",
             "url": f"{_BOOKS_URL}/%D8%A7%D9%84%D9%85%D8%B3%D8%B1%D8%AD_%D9%88%D8%A7%D9%84%D8%B9%D8%A7%D9%84%D9%85_%D8%A7%D9%84%D8%B1%D9%82%D9%85%D9%8A.epub",
             "sha256": "31565f651e4d07cc97aacfb2c270164b5813f419fe21a0f600e8e86e35b05ba2"},
            {"file": "data_science.epub", "title": "علم البيانات", "author": "جون دي كيليهر وبريندان تيرني",
             "rights": "Hindawi (design, Arabic translation); original text © 2018 MIT",
             "url": f"{_BOOKS_URL}/%D8%B9%D9%84%D9%85_%D8%A7%D9%84%D8%A8%D9%8A%D8%A7%D9%86%D8%A7%D8%AA.epub",
             "sha256": "cdc651aeafef692740015bbcebd5f3fd6107a83b9090219f16b8931da76ddf88"},
        ],
    },
]

# Batch 2 (commit 06a24b0): only the books approved after data.books_candidate (clean text, clear license).
# The cleaned text comes from data/processed/private_books_v2_candidate/books/<name>.txt; data.add_books
# adds them to a copy of dataset v2. "weight": "match:<source>" gives the book the same exposure per
# character as that source; a number is a fixed sampling weight. held_out=False: too small to split.
_BATCH2_COMMIT = "06a24b06e9529d0e2a7186611c93bc9143f25f0c"
_BATCH2_URL = f"https://media.githubusercontent.com/media/hishamhk12/HishamLM-v0.1/{_BATCH2_COMMIT}/data/private_books/raw/ar"
_WIKISOURCE = {"license": "CC-BY-SA-3.0 / GFDL (Wikisource export; the original work is in the public domain)",
               "license_url": "https://creativecommons.org/licenses/by-sa/3.0/", "private": False}

BOOKS_BATCH2 = [
    {"name": "kalila_wa_dimna", "language": "ar", "eval_group": "books_new_kalila", "weight": "match:hindawi_books_ar",
     "held_out": True, "title": "كليلة ودمنة", "author": "ابن المقفع", "file": "كليلة_ودمنة.epub",
     "url": f"{_BATCH2_URL}/%D9%83%D9%84%D9%8A%D9%84%D8%A9_%D9%88%D8%AF%D9%85%D9%86%D8%A9.epub",
     "sha256": "2197f50f483a01346ad438ee8c09fc5bee9b5fe53dc299bf9e8e4d2d72aea475", **_WIKISOURCE},
    {"name": "nazarat_al_bakhil", "language": "ar", "eval_group": "books_new_nazarat", "weight": "match:hindawi_books_ar",
     "held_out": False, "title": "النظرات: البخيل", "author": "مصطفى لطفي المنفلوطي", "file": "النظرات_البخيل.epub",
     "url": f"{_BATCH2_URL}/%D8%A7%D9%84%D9%86%D8%B8%D8%B1%D8%A7%D8%AA_%D8%A7%D9%84%D8%A8%D8%AE%D9%8A%D9%84.epub",
     "sha256": "d31a96aa5d7f1ab13ca0fb561d030eb0fcbc7ce617f8223d424ade3dc77e3ef3", **_WIKISOURCE},
    {"name": "al_bayan_wal_tabyin", "language": "ar", "eval_group": "books_new_bayan", "weight": 0.01,
     "held_out": True, "title": "البيان والتبيين", "author": "الجاحظ", "file": "البيان_والتبيين.epub",
     "url": f"{_BATCH2_URL}/%D8%A7%D9%84%D8%A8%D9%8A%D8%A7%D9%86_%D9%88%D8%A7%D9%84%D8%AA%D8%A8%D9%8A%D9%8A%D9%86.epub",
     "sha256": "957e24256045b6f8821d336705435948e302c1203da2256bcb760fb9ddd80fc6", **_WIKISOURCE},
]

REJECTED = [
    {"name": "المعجم الوسيط / معجم اللغة العربية المعاصرة (scanned PDFs)", "reason": "Image-only scans (no text "
     "layer, would need OCR); dictionary entries rather than prose. Removed from the repository by the owner."},
    {"name": "Smith & Van Dyck Arabic Bible (1865)", "reason": "Public domain, but classical religious language; "
     "kept out by decision (possible secondary source later)."},
    {"name": "UD Arabic-PUD treebank", "reason": "README says sentences 'may be copyrighted in some jurisdictions' "
     "and were collected under copyright exceptions: license not clear enough."},
    {"name": "ARCD (Arabic Reading Comprehension Dataset)", "reason": "Repository is MIT for code; no explicit license "
     "for the Wikipedia paragraphs it contains."},
    {"name": "thiagobodruk/bible ar_svd.json", "reason": "Repository states 'all rights reserved to the owners'."},
    {"name": "Common Voice Arabic classical lines", "reason": "Quranic verses, classical poetry and proverbs in the same "
     "file are filtered out: not modern Arabic."},
    {"name": "Gutenberg KJV / Milton / Shakespeare / Blake / Whitman", "reason": "Public domain but archaic or verse; "
     "not representative of modern English."},
    {"name": "NTREX English source side (WMT19 news)", "reason": "Original news articles; license of the source text unclear."},
    {"name": "Batch 2: بداية المجتهد 1 و 2، الخيميائي، دماغك تحت تأثير الإباحية", "reason": "Internet Archive OCR with "
     "heavy noise (grade C in data.books_candidate); the two translations also have unclear rights."},
    {"name": "Batch 2: adab.epub", "reason": "OCR output is random Latin text (Internet Archive estimates 0.4-5% accuracy)."},
    {"name": "Batch 2: العقد الفريد، مجلة الرسالة (Wikisource)", "reason": "The exports contain only index pages, no text."},
    {"name": "Batch 2: الجريمة والعقاب (2)، ترجمة سامي الدروبي", "reason": "Clean text, but the rights to the Arabic "
     "translation are not confirmed yet; postponed by the project owner."},
    {"name": "Wikipedia dumps, Hindawi, OpenITI, Tatoeba, OPUS, FLORES, TyDi QA", "reason": "Blocked from this "
     "environment's network, or non-commercial license (OpenITI: CC BY-NC-SA)."},
]


def get_source(name):
    for source in SOURCES + PRIVATE_SOURCES + BOOKS_BATCH2:
        if source["name"] == name:
            return source
    raise KeyError(name)
