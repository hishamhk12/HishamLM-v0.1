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

REJECTED = [
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
    {"name": "Wikipedia dumps, Hindawi, OpenITI, Tatoeba, OPUS, FLORES, TyDi QA", "reason": "Blocked from this "
     "environment's network, or non-commercial license (OpenITI: CC BY-NC-SA)."},
]


def get_source(name):
    for source in SOURCES:
        if source["name"] == name:
            return source
    raise KeyError(name)
