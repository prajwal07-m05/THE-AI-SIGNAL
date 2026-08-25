"""Entity-resolution tests — the assignment's canonical example + edge cases."""
from src.resolver.entity_resolver import EntityResolver, normalize


def test_openai_variants_collapse_to_canonical():
    r = EntityResolver()
    for variant in ["OpenAI", "OpenAI, Inc.", "Open AI", "openai llc", "OPENAI"]:
        assert r.resolve(variant).canonical == "OpenAI"


def test_alias_and_fuzzy():
    r = EntityResolver()
    assert r.resolve("deepmind").canonical == "Google DeepMind"
    assert r.resolve("hugging face inc").canonical == "Hugging Face"
    # typo -> fuzzy match
    assert r.resolve("Anthropicc").canonical == "Anthropic"


def test_unknown_mints_new_and_is_stable():
    r = EntityResolver()
    a = r.resolve("Zyxware Quantum Labs")
    b = r.resolve("zyxware quantum")  # fuzzy hit against the freshly-minted canonical
    assert a.method == "new"
    assert b.canonical == a.canonical


def test_normalize_strips_legal_suffixes():
    assert normalize("Cohere Technologies, Inc.") == "cohere"


def test_mapping_log_records_every_decision():
    r = EntityResolver()
    r.resolve("Open AI")
    r.resolve("Scale AI, Inc.")
    assert len(r.log) == 2
    assert {m.method for m in r.log} <= {"exact", "alias", "fuzzy", "new"}
