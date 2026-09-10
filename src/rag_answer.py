import re

from retrieve import load_resources, retrieve


# --------------------------------------------------
# Configuration
# --------------------------------------------------

WASTE_CLASSES = [
    "glass",
    "leather",
    "organic",
    "paper",
    "plastic"
]

CLASS_ALIASES = {
    "paper": ["paper", "cardboard", "newspaper"],
    "glass": ["glass"],
    "plastic": ["plastic", "pet"],
    "organic": [
        "organic",
        "food waste",
        "food scraps",
        "kitchen waste",
        "compost",
        "composting",
        "biodegradable",
        "yard waste",
        "yard trimmings",
        "wet waste"
    ],
    "leather": ["leather", "rexine"],
}

CLASS_QUERY_HINTS = {
    "paper": "paper cardboard dry recyclable waste rag-pickers",
    "glass": "glass bottles recycling bin broken glass separated at source",
    "plastic": "plastic PET bottles plastic bags dry recyclable waste",
    "organic": "organic compost food scraps wet biodegradable waste",
    "leather": "leather rags rexine recyclable waste sold to raddiwala scrap dealer",
}

FINAL_K = 5
WINDOW_SENTENCES = 3

CLASS_SPECIFIC_TERMS = {
    "paper": ["cardboard", "newspaper"],
    "glass": ["glass bottles", "glass containers", "broken glass"],
    "plastic": [
        "plastic bags",
        "plastic bottles",
        "plastic containers",
        "pet"
    ],
    "organic": [
        "food scraps",
        "compost pit",
        "wet biodegradable",
        "yard trimmings"
    ],
    "leather": ["rexine"],
}

RELATED_TERMS = {
    "paper": ["dry waste", "recyclable waste"],
    "glass": ["dry waste", "recyclable waste"],
    "plastic": ["dry waste", "recyclable waste"],
    "organic": ["wet waste", "green container"],
    "leather": ["dry waste", "recyclable waste"],
}

STRONG_HANDLING_ACTIONS = [
    "handed over",
    "hand over",
    "segregate",
    "segregated",
    "segregation",
    "separate",
    "separated",
    "separation",
    "sold to",
    "give to",
    "given to",
    "compost",
    "composted",
    "composting",
    "should not",
    "do not",
    "put it in",
    "collected in",
    "scrap dealer",
    "raddiwala",
    "kabari",
    "rag-picker",
]

RECYCLE_ACTIONS = [
    "recycle",
    "recycled",
    "recycling",
    "reuse",
    "reused",
    "accepted",
    "accept",
]

HANDLING_ACTIONS = STRONG_HANDLING_ACTIONS + RECYCLE_ACTIONS + [
    "dispose",
    "disposal",
    "container",
    "bin",
]

BACKGROUND_MARKERS = [
    "percent",
    "statistics",
    "in 2018",
    "in 2019",
    "in 2020",
    "saves trees",
    "natural resources",
    "makes up",
    "generated each year",
    "million tons",
    "fourth tier",
    "wasted food scale",
    "click to enlarge",
    "form of organics recycling",
    "only 5%",
]

INSTRUCTIONAL_PHRASES = [
    "should",
    "can be",
    "handed over",
    "sold to",
    "put it in",
    "should not",
    "do not",
    "must",
]

COLLECTION_ACTIONS = [
    "handed over",
    "hand over",
    "give to",
    "given to",
    "sold to",
    "collected in",
    "segregate",
    "segregated",
    "segregation",
]

DISPOSAL_WARNING_MARKERS = [
    "should not be dumped",
    "do not dump",
    "dumped into municipal",
    "do not litter",
    "not be dumped",
]

INSUFFICIENT_EVIDENCE = (
    "The available sources do not provide sufficient information "
    "to answer this question."
)

MIN_SCORE = 36

MATERIAL_CLASSES = [
    "paper",
    "glass",
    "plastic",
    "leather",
]


def detect_waste_class(question):

    question_lower = question.lower()

    for waste_class in WASTE_CLASSES:

        if waste_class in question_lower:
            return waste_class

    return None


def detect_question_intent(question):

    question_lower = question.lower()

    if any(
        term in question_lower
        for term in ["recycl", "reuse"]
    ):
        return "recycle"

    if "compost" in question_lower:
        return "compost"

    return "handle"


def format_retrieved_sources(results):

    best_scores = {}
    source_order = []

    for result in results:
        source_id = result["source_id"]
        score = result["score"]

        if source_id not in best_scores:
            source_order.append(source_id)
            best_scores[source_id] = score
        elif score > best_scores[source_id]:
            best_scores[source_id] = score

    return [
        f"{source_id} ({best_scores[source_id]:.4f})"
        for source_id in source_order
    ]


def _merge_results(*result_lists):

    merged = {}

    for results in result_lists:
        for result in results:
            chunk_id = result["chunk_id"]
            current = merged.get(chunk_id)
            if current is None or result["score"] > current["score"]:
                merged[chunk_id] = result

    return sorted(
        merged.values(),
        key=lambda item: item["score"],
        reverse=True
    )


def gather_evidence(question, model, index, metadata, waste_class=None):

    if waste_class is None:
        waste_class = detect_waste_class(question)
    else:
        waste_class = str(waste_class).lower()

    results = retrieve(question, model, index, metadata)

    hint = CLASS_QUERY_HINTS.get(waste_class)
    if hint:
        results = _merge_results(
            results,
            retrieve(hint, model, index, metadata)
        )

    if waste_class:
        matching = [
            result for result in results
            if _mentions_class(result["text"].lower(), waste_class)
        ]
        if matching:
            results = matching

    return results[:FINAL_K]


def _contains_term(text, term):

    if term == "%":
        return "%" in text

    if term == "recyclable waste":
        return re.search(
            r"(?<!non-)(?<!non )\brecyclable waste\b",
            text
        ) is not None

    return re.search(
        r"\b" + re.escape(term) + r"\b",
        text
    ) is not None


def _contains_any(text, terms):

    return any(_contains_term(text, term) for term in terms)


def _normalize_class_text(text, waste_class):

    if waste_class == "paper":
        text = text.replace("carbon paper", " ")

    return text


def _mentions_class(text, waste_class):

    if not waste_class:
        return False

    text = _normalize_class_text(text, waste_class)

    return _contains_any(
        text,
        CLASS_ALIASES.get(waste_class, [waste_class])
    )


def _mentions_other_class(text, waste_class):

    for other_class in MATERIAL_CLASSES:

        if other_class == waste_class:
            continue

        if _mentions_class(text, other_class) and not _mentions_class(
            text,
            waste_class
        ):
            return True

    return False


def _other_material_count(text, waste_class):

    count = 0

    for other_class in MATERIAL_CLASSES:

        if other_class == waste_class:
            continue

        if _mentions_class(text, other_class):
            count += 1

    return count


def _asks_for_statistics(question):

    question_lower = question.lower()

    return any(
        term in question_lower
        for term in ["percent", "percentage", "statistic", "how much"]
    )


def _is_background_statistic(text):

    if re.search(r"\d+\s*%", text):
        return True

    return _contains_any(text, BACKGROUND_MARKERS)


def _strip_yes_no_prefix(sentence):

    match = re.match(
        r"^(yes|no)[,.]?\s+",
        sentence,
        flags=re.IGNORECASE
    )

    if not match:
        return sentence, False

    remainder = sentence[match.end():].strip()

    if not remainder:
        return sentence, True

    remainder = remainder[0].upper() + remainder[1:]

    return remainder, True


def _clean_chunk_text(text):

    text = re.sub(r"--- PAGE \d+ ---", " ", text)
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(
        r"and\s*despatched",
        "and despatched",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r"anddespatched",
        "and despatched",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r"rag-?\s*pickers",
        "rag-pickers",
        text,
        flags=re.IGNORECASE
    )

    kept_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if re.match(r"^z\s+\S.{0,40}$", line) and not re.search(
            r"[.!?]$",
            line
        ):
            continue

        if (
            re.match(r"^\d+\.\d+\s+\S", line)
            and len(line) < 90
            and not re.search(r"[.!?]$", line)
        ):
            continue

        line = re.sub(r"^z\s+", "", line)
        kept_lines.append(line)

    text = " ".join(kept_lines)
    text = re.sub(
        r"\)\s+(The|It|Waste|Dry|This)\b",
        r"). \1",
        text
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _split_sentences(text):

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _trim_long_sentence(sentence, waste_class):

    if len(sentence) <= 350:
        return sentence

    if not waste_class:
        return sentence[:350]

    match = re.search(
        r"[^.]*"
        + re.escape(waste_class)
        + r"[^.]*"
        r"(?:recycle|reuse|handed over|segregat|separate|sold to|accept)[^.]*\.",
        sentence.lower()
    )

    if match:
        return sentence[match.start():match.end()]

    return sentence[:350]


def _score_sentence(
    sentence,
    waste_class,
    source_id,
    question,
    window_text="",
    chunk_mentions_class=False
):

    sentence, _had_yes_no = _strip_yes_no_prefix(sentence)
    sentence = re.sub(r"^[\s.]+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence_lower = sentence.lower()
    window_lower = (window_text or sentence).lower()

    if len(sentence) < 40:
        return None

    if not re.search(r"[.!?]$", sentence):
        return None

    if sentence_lower.endswith("?"):
        return None

    if sentence[0].islower() or sentence[0] in ",;:–-(":
        return None

    if re.match(r"^\d+\s+", sentence):
        return None

    if _contains_any(
        sentence_lower,
        [
            "learn more",
            "frequently asked",
            "click to enlarge",
            "on this page",
            "given below",
            "thermocol",
        ]
    ):
        return None

    if sentence.count(",") >= 6:
        return None

    allow_stats = _asks_for_statistics(question)

    if not allow_stats and _is_background_statistic(sentence_lower):
        return None

    has_strong = _contains_any(sentence_lower, STRONG_HANDLING_ACTIONS)
    has_recycle = _contains_any(sentence_lower, RECYCLE_ACTIONS)
    if _contains_any(
        sentence_lower,
        [
            "difficult to recycle",
            "hard to recycle",
            "cannot be recycled",
            "phased out",
            "eliminated",
        ]
    ):
        has_recycle = False
    has_handling = has_strong or has_recycle or _contains_any(
        sentence_lower,
        HANDLING_ACTIONS
    )

    if not has_handling:
        return None

    mentions_class = _mentions_class(sentence_lower, waste_class)
    mentions_class_window = _mentions_class(window_lower, waste_class)
    related_in_sentence = _contains_any(
        sentence_lower,
        RELATED_TERMS.get(waste_class, [])
    )
    context_supported = (
        (not mentions_class)
        and (
            mentions_class_window
            or (chunk_mentions_class and related_in_sentence)
        )
    )

    if waste_class and not mentions_class and not context_supported:
        return None

    if (
        waste_class == "organic"
        and has_recycle
        and not _contains_any(
            sentence_lower,
            ["compost", "composting", "food scraps", "biodegradable"]
        )
    ):
        return None

    if (
        waste_class != "organic"
        and _contains_any(
            sentence_lower,
            [
                "biodegradable waste",
                "biodegradable wet",
                "organic waste",
                "wet waste",
                "composting",
                "food scraps",
            ]
        )
        and not mentions_class
    ):
        return None

    score = 0

    if mentions_class:
        score += 30
    elif context_supported:
        score += 20

    if has_strong:
        score += 10

    if _contains_any(sentence_lower, INSTRUCTIONAL_PHRASES):
        score += 8

    if has_recycle:
        score += 6

    intent = detect_question_intent(question)
    matches_intent = False

    if intent == "recycle":
        if has_recycle or _contains_any(sentence_lower, COLLECTION_ACTIONS):
            score += 12
            matches_intent = True
        elif _contains_any(sentence_lower, DISPOSAL_WARNING_MARKERS):
            score -= 20
    elif intent == "compost":
        if _contains_any(
            sentence_lower,
            ["compost", "composting", "biodegradable"]
        ):
            score += 12
            matches_intent = True
    else:
        matches_intent = has_handling

    if _contains_any(
        sentence_lower,
        CLASS_SPECIFIC_TERMS.get(waste_class, [])
    ):
        score += 5

    if _contains_any(
        sentence_lower,
        RELATED_TERMS.get(waste_class, [])
    ):
        score += 1

    if source_id == "S2":
        score += 2
    elif source_id == "S4" and waste_class == "organic":
        score += 2
    elif source_id == "S3" and waste_class in ("glass", "plastic", "paper"):
        score += 1

    if _mentions_other_class(sentence_lower, waste_class):
        score -= 15

    class_focused = _other_material_count(sentence_lower, waste_class) < 2

    if not class_focused:
        score -= 15

    if _contains_any(
        sentence_lower,
        ["when you shop", "look for products"]
    ):
        score -= 15

    if (
        waste_class == "plastic"
        and "non-recyclable" in sentence_lower
        and "recycl" in question.lower()
    ):
        return None

    if score < MIN_SCORE:
        return None

    return {
        "score": score,
        "sentence": sentence,
        "source_id": source_id,
        "has_strong": has_strong,
        "context_supported": context_supported,
        "matches_intent": matches_intent,
        "class_focused": class_focused,
    }


def select_answer_sentences(question, results):

    if not results:
        return []

    waste_class = detect_waste_class(question)
    candidates = []
    seen_sentences = set()

    for result in results:

        source_id = result["source_id"]
        text = _clean_chunk_text(result["text"])
        chunk_mentions_class = _mentions_class(text.lower(), waste_class)
        sentences = [
            _trim_long_sentence(sentence, waste_class)
            for sentence in _split_sentences(text)
        ]

        for index, sentence in enumerate(sentences):

            window_start = max(0, index - WINDOW_SENTENCES)
            window_text = " ".join(sentences[window_start:index + 1])
            scored = _score_sentence(
                sentence,
                waste_class,
                source_id,
                question,
                window_text=window_text,
                chunk_mentions_class=chunk_mentions_class
            )

            if scored is None:
                continue

            key = scored["sentence"].lower()
            if key in seen_sentences:
                continue

            seen_sentences.add(key)
            candidates.append(scored)

    if not candidates:
        return []

    intent = detect_question_intent(question)

    focused_candidates = [
        item for item in candidates if item.get("class_focused")
    ]

    if focused_candidates:
        candidates = focused_candidates

    if intent == "recycle":
        intent_candidates = [
            item for item in candidates if item.get("matches_intent")
        ]
        if intent_candidates:
            candidates = intent_candidates

    explicit_candidates = [
        item for item in candidates
        if not item.get("context_supported")
    ]

    if explicit_candidates:
        candidates = explicit_candidates

    strong_candidates = [
        item for item in candidates if item["has_strong"]
    ]

    if strong_candidates:
        candidates = strong_candidates

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return candidates[:2]


def generate_answer(question, results, tokenizer=None, model=None):

    selected = select_answer_sentences(question, results)

    if not selected:
        return INSUFFICIENT_EVIDENCE

    waste_class = detect_waste_class(question)
    answer_parts = []
    added_class_lead = False

    for item in selected:

        sentence = item["sentence"]

        if (
            waste_class
            and item.get("context_supported")
            and not _mentions_class(sentence.lower(), waste_class)
            and not added_class_lead
        ):
            class_label = waste_class.capitalize()

            if _contains_term(sentence.lower(), "recyclable waste"):
                sentence = (
                    f"{class_label} is listed among recyclable waste "
                    f"in the provided guidelines. {sentence}"
                )
            else:
                sentence = (
                    f"{class_label} is listed among the waste categories "
                    f"in the provided guidelines. {sentence}"
                )

            added_class_lead = True

        answer_parts.append(sentence)

    answer = " ".join(answer_parts)

    selected_sources = []
    for item in selected:
        if item["source_id"] not in selected_sources:
            selected_sources.append(item["source_id"])

    citations = "".join(
        f"[{source_id}]"
        for source_id in selected_sources
    )

    return f"{answer} {citations}"


def main():

    print("=" * 70)
    print("GROUNDED RAG ANSWER GENERATION")
    print("=" * 70)

    print("\nLoading retrieval resources...")

    retrieval_model, index, metadata = load_resources()

    print(
        f"Indexed chunks: {index.ntotal}"
    )

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("No question provided.")

        return

    print("\nRetrieving evidence...")

    results = gather_evidence(
        question,
        retrieval_model,
        index,
        metadata
    )

    print("\nRetrieved sources:")

    displayed_sources = format_retrieved_sources(results)

    for rank, source_text in enumerate(
        displayed_sources,
        start=1
    ):

        print(
            f"{rank}. {source_text}"
        )

    selected = select_answer_sentences(question, results)

    print("\nSelected evidence:")

    if not selected:
        print("None")
    else:
        for item in selected:
            print(
                f"- ({item['score']}) [{item['source_id']}] "
                f"{item['sentence']}"
            )

    print("\nGenerating grounded answer...")

    answer = generate_answer(question, results)

    print("\n" + "=" * 70)
    print("GROUNDED ANSWER")
    print("=" * 70)

    print(answer)


if __name__ == "__main__":
    main()
