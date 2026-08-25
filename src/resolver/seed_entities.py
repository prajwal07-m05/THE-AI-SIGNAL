"""Seed database of known canonical AI entities (Phase IV).

A curated list of 50 well-known AI companies. Extracted messy strings are mapped
against these canonical forms. In production this table lives in Postgres and is
grown continuously; here it is a mock as the assignment permits.

`aliases` are common surface forms we KNOW map to the canonical name — used for
exact/alias hits before we fall back to fuzzy matching.
"""
from __future__ import annotations

CANONICAL_ENTITIES: dict[str, list[str]] = {
    "OpenAI": ["openai inc", "open ai", "openai, inc.", "openai llc"],
    "Anthropic": ["anthropic pbc", "anthropic ai"],
    "Google DeepMind": ["deepmind", "google deepmind technologies", "deep mind"],
    "Meta AI": ["facebook ai research", "fair", "meta platforms ai"],
    "Mistral AI": ["mistral", "mistralai"],
    "Cohere": ["cohere inc", "cohere ai", "cohere technologies"],
    "Hugging Face": ["huggingface", "hugging face inc"],
    "Stability AI": ["stability", "stabilityai", "stability ai ltd"],
    "Perplexity AI": ["perplexity", "perplexity labs"],
    "xAI": ["x ai", "x.ai"],
    "Databricks": ["databricks inc", "data bricks"],
    "Scale AI": ["scale", "scale ai inc", "scaleai"],
    "Runway": ["runway ml", "runwayml", "runway ai"],
    "Character.AI": ["character ai", "characterai", "c.ai"],
    "Inflection AI": ["inflection", "inflection ai inc"],
    "Adept AI": ["adept", "adept ai labs"],
    "Together AI": ["together", "together computer", "togetherai"],
    "Groq": ["groq inc"],
    "Cerebras Systems": ["cerebras"],
    "Weights & Biases": ["wandb", "weights and biases", "w&b"],
    "LangChain": ["langchain inc", "lang chain"],
    "LlamaIndex": ["llama index", "gpt index"],
    "Pinecone": ["pinecone systems", "pinecone io"],
    "Weaviate": ["weaviate b.v.", "semi technologies"],
    "Chroma": ["chromadb", "chroma inc"],
    "Replicate": ["replicate inc"],
    "Modal": ["modal labs", "modal com"],
    "Fireworks AI": ["fireworks", "fireworks ai inc"],
    "Anyscale": ["any scale", "anyscale inc"],
    "AssemblyAI": ["assembly ai", "assembly"],
    "ElevenLabs": ["eleven labs", "11labs"],
    "Synthesia": ["synthesia ltd"],
    "Jasper": ["jasper ai", "jasper.ai"],
    "Copy.ai": ["copy ai", "copyai"],
    "Glean": ["glean ai", "glean technologies"],
    "Harvey": ["harvey ai", "harvey.ai"],
    "Cursor": ["anysphere", "cursor ai", "cursor.so"],
    "Codeium": ["codeium inc", "exafunction"],
    "Tabnine": ["tab nine", "tabnine ltd"],
    "Midjourney": ["mid journey", "midjourney inc"],
    "Notion": ["notion labs", "notion ai"],
    "Grammarly": ["grammarly inc"],
    "DataRobot": ["data robot"],
    "H2O.ai": ["h2o ai", "h2o", "h2oai"],
    "Snowflake": ["snowflake inc", "snowflake computing"],
    "NVIDIA": ["nvidia corporation", "nvidia corp"],
    "IBM Watson": ["watson", "ibm watson ai"],
    "AI21 Labs": ["ai21", "ai 21 labs"],
    "Aleph Alpha": ["alephalpha"],
    "Contextual AI": ["contextual", "contextual ai inc"],
}
