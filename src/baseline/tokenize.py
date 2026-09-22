"""Tokenization with the Assignment 1 stopword list."""

from __future__ import annotations

import re

STOPWORDS = set("""a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she
she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves""".split())

# letter runs, hyphen-joined allowed
WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")


def tokenize(text: str) -> list[str]:
    tokens = []
    for w in WORD_RE.findall(text.lower()):
        if len(w) < 2:  # drop single letters (math variables)
            continue
        if w in STOPWORDS:
            continue
        tokens.append(w)
    return tokens


def add_terms_to_docs(docs: dict[int, dict]) -> dict[int, dict]:
    """Add deduped `terms` per doc, keeping the Assignment 1 docs shape.

    NOTE: dedupes to a set (like Assignment 1). build_doc_term_freqs in
    tfidf.py re-tokenizes the abstract, so per-occurrence counts stay intact.
    """
    for doc_id, doc in docs.items():
        doc["terms"] = list(set(tokenize(doc["abstract"])))
    return docs


def unique_terms(docs: dict[int, dict]) -> set[str]:
    vocab = set()
    for doc in docs.values():
        vocab.update(doc["terms"])
    return vocab
