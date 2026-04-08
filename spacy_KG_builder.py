"""
spaCy → LightRAG KG Builder  (English, en_core_web_lg)
=======================================================
Uses spaCy for NER + dependency-based relation extraction,
assembles results into LightRAG's custom_kg format, then
inserts via insert_custom_kg — no LLM needed at index time.

Setup:
    pip install spacy lightrag-hku
    python -m spacy download en_core_web_lg
"""

import hashlib
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional
import spacy
from spacy.tokens import Doc, Span


# ─────────────────────────────────────────────
# 1. Data structures
# ─────────────────────────────────────────────

@dataclass
class KGEntity:
    entity_name: str
    entity_type: str
    description: str
    source_id: str
    file_path: str = "unknown"


@dataclass
class KGRelationship:
    src_id: str
    tgt_id: str
    description: str
    keywords: str
    weight: float
    source_id: str
    file_path: str = "unknown"


@dataclass
class KGChunk:
    content: str
    source_id: str
    file_path: str = "unknown"


@dataclass
class CustomKG:
    chunks: list[KGChunk] = field(default_factory=list)
    entities: list[KGEntity] = field(default_factory=list)
    relationships: list[KGRelationship] = field(default_factory=list)

    def to_lightrag_format(self) -> dict:
        """Convert to the dict format expected by rag.insert_custom_kg()."""
        return {
            "chunks": [
                {
                    "content": c.content,
                    "source_id": c.source_id,
                    "file_path": c.file_path,
                }
                for c in self.chunks
            ],
            "entities": [
                {
                    "entity_name": e.entity_name,
                    "entity_type": e.entity_type,
                    "description": e.description,
                    "source_id": e.source_id,
                    "file_path": e.file_path,
                }
                for e in self.entities
            ],
            "relationships": [
                {
                    "src_id": r.src_id,
                    "tgt_id": r.tgt_id,
                    "description": r.description,
                    "keywords": r.keywords,
                    "weight": r.weight,
                    "source_id": r.source_id,
                    "file_path": r.file_path,
                }
                for r in self.relationships
            ],
        }

    def print_summary(self) -> None:
        """Print a readable summary — useful for verifying results."""
        print(f"\n{'='*60}")
        print(f"  Chunks       : {len(self.chunks)}")
        print(f"  Entities     : {len(self.entities)}")
        print(f"  Relationships: {len(self.relationships)}")
        print(f"{'='*60}")

        print("\n── Entities ──────────────────────────────────────────")
        for e in self.entities:
            print(f"  [{e.entity_type:14s}]  {e.entity_name}")
            print(f"               {e.description[:90]}")

        print("\n── Relationships ─────────────────────────────────────")
        for r in self.relationships:
            print(f"  {r.src_id}  --[w={r.weight:.1f}]-->  {r.tgt_id}")
            print(f"    keywords : {r.keywords}")
            print(f"    desc     : {r.description[:80]}")


# ─────────────────────────────────────────────
# 2. Entity type map  (en_core_web_lg labels)
# ─────────────────────────────────────────────
#
# en_core_web_lg uses OntoNotes 5 labels.
# Labels not listed here are silently ignored.
#
# Full label reference:
#   PERSON      People, including fictional
#   NORP        Nationalities, religious/political groups
#   FAC         Buildings, airports, highways, bridges
#   ORG         Companies, agencies, institutions
#   GPE         Countries, cities, states
#   LOC         Non-GPE locations (mountains, bodies of water)
#   PRODUCT     Objects, vehicles, foods (not services)
#   EVENT       Named hurricanes, battles, wars, sports events
#   WORK_OF_ART Titles of books, songs, films
#   LAW         Named documents made into laws
#   LANGUAGE    Any named language
#   DATE        Absolute or relative dates / periods
#   TIME        Times smaller than a day
#   MONEY       Monetary values (commented out below — often noisy)
#   QUANTITY    Measurements (commented out — often noisy)

EN_ENTITY_TYPE_MAP: dict[str, str] = {
    "PERSON":      "person",
    "NORP":        "organization",
    "FAC":         "location",
    "ORG":         "organization",
    "GPE":         "location",
    "LOC":         "location",
    "PRODUCT":     "product",
    "EVENT":       "event",
    "WORK_OF_ART": "concept",
    "LAW":         "concept",
    "LANGUAGE":    "concept",
    "DATE":        "time",
    "TIME":        "time",
    # Uncomment if you want monetary / measurement entities:
    # "MONEY":    "concept",
    # "QUANTITY": "concept",
    # "PERCENT":  "concept",
}


# ─────────────────────────────────────────────
# 3. Core builder
# ─────────────────────────────────────────────

class SpacyKGBuilder:
    """
    Builds a LightRAG-compatible KG from plain text using spaCy.

    Pipeline per chunk:
        NER  ->  collect entities + context sentences
        Dependency parse  ->  SVO relation extraction
        Co-occurrence window  ->  weak relation fallback

    Then across all chunks:
        filter low-frequency entities
        merge duplicate relation edges (accumulate weight)
        assemble CustomKG
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        entity_type_map: Optional[dict[str, str]] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 50,
        coref_window: int = 3,
        min_entity_freq: int = 1,
        dep_relation_weight: float = 1.0,
        cooc_relation_weight: float = 0.5,
    ):
        """
        Args:
            model_name           : spaCy model to load
            entity_type_map      : override the default EN_ENTITY_TYPE_MAP
            chunk_size           : characters per chunk (default 600)
            chunk_overlap        : overlap between adjacent chunks (default 80)
            coref_window         : sentence window for co-occurrence (default 3)
            min_entity_freq      : drop entities seen fewer than this many times
            dep_relation_weight  : weight assigned to SVO-derived edges
            cooc_relation_weight : weight assigned to co-occurrence edges
        """
        print(f"Loading spaCy model '{model_name}' ...")
        self.nlp = spacy.load(model_name)
        self.entity_type_map = entity_type_map or EN_ENTITY_TYPE_MAP
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.coref_window = coref_window
        self.min_entity_freq = min_entity_freq
        self.dep_relation_weight = dep_relation_weight
        self.cooc_relation_weight = cooc_relation_weight
        print("Model loaded.")

    # ── Text splitting ────────────────────────

    def _presplit_text(self, text: str, max_chars: int = 900_000) -> list[str]:
        """
        Split text into segments each shorter than max_chars before passing
        to spaCy, breaking at paragraph → sentence → newline → hard-cut order.
        This is needed for very long documents that exceed nlp.max_length.
        """
        if len(text) <= max_chars:
            return [text]

        segments: list[str] = []
        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                seg = text[start:].strip()
                if seg:
                    segments.append(seg)
                break
            # Prefer splitting at paragraph break
            split = text.rfind('\n\n', start, end)
            if split <= start:
                # Fall back to sentence end
                split = text.rfind('. ', start, end)
            if split <= start:
                split = text.rfind('\n', start, end)
            if split <= start:
                split = end          # hard cut as last resort
            else:
                split += 1           # include the delimiter
            seg = text[start:split].strip()
            if seg:
                segments.append(seg)
            start = split
        return segments

    def _split_into_sentence_chunks(
        self, doc, file_path: str
    ) -> tuple[list[KGChunk], list[tuple[int, int]], list[str]]:
        """
        Split an already-parsed spaCy Doc into sentence-boundary-aligned chunks.

        Returns:
            chunks     : list of KGChunk, each with a unique content-based source_id
            spans      : list of (tok_start, tok_end) absolute token indices
                         corresponding to each chunk in the parent Doc
            source_ids : per-chunk source_id strings (parallel to chunks/spans)
        """
        sentences = list(doc.sents)
        if not sentences:
            return [], [], []

        chunks:     list[KGChunk] = []
        spans:      list[tuple[int, int]] = []
        source_ids: list[str] = []
        i = 0

        while i < len(sentences):
            group: list = []
            total_chars = 0

            for j in range(i, len(sentences)):
                sent_len = len(sentences[j].text)
                # Always include at least one sentence per chunk
                if total_chars + sent_len > self.chunk_size and group:
                    break
                group.append(sentences[j])
                total_chars += sent_len

            if not group:           # single sentence longer than chunk_size
                group = [sentences[i]]

            chunk_text = " ".join(s.text.strip() for s in group)
            tok_start  = group[0].start
            tok_end    = group[-1].end

            chunk_source_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            chunks.append(KGChunk(content=chunk_text, source_id=chunk_source_id, file_path=file_path))
            spans.append((tok_start, tok_end))
            source_ids.append(chunk_source_id)

            # Overlap: count trailing sentences whose total length ≤ chunk_overlap
            overlap_count = 0
            overlap_chars = 0
            for s in reversed(group):
                if overlap_chars + len(s.text) > self.chunk_overlap:
                    break
                overlap_chars += len(s.text)
                overlap_count += 1

            advance = max(1, len(group) - overlap_count)
            i += advance

        return chunks, spans, source_ids

    # ── Description builders ──────────────────

    def _make_entity_description(
        self, name: str, etype: str, contexts: list[str]
    ) -> str:
        ctx = "; ".join(s.strip() for s in contexts[:3] if s.strip())
        if ctx:
            return f"{name} is a {etype}. Context: {ctx}"
        return f"{name} is a {etype}."

    def _make_relation_description(
        self, src: str, tgt: str, verb_lemma: str, sentence: str
    ) -> tuple[str, str]:
        via = f' via "{verb_lemma}"' if verb_lemma and verb_lemma != "co-occurrence" else ""
        desc = f"{src} and {tgt} are related{via}. Source: {sentence[:100].strip()}"
        keywords = ",".join(k for k in [src, tgt, verb_lemma] if k)
        return desc, keywords

    # ── Dependency-based SVO extraction ───────

    def _find_entity_for_token(
        self, token, sent_ents: list[Span]
    ) -> Optional[str]:
        for ent in sent_ents:
            if ent.start <= token.i < ent.end:
                return ent.text.strip().lower()   # normalized
        return None

    def _extract_dep_relations(
        self,
        doc,
        entity_names: set[str],
        source_id: str,
        file_path: str,
    ) -> list[KGRelationship]:
        """
        Extract Subject-Verb-Object triples where both subject and object
        are known named entities.

        Dependency labels covered:
            nsubj / nsubjpass  -- nominal subject
            dobj / obj         -- direct object
            attr               -- attribute  (X is Y)
            pobj               -- object of preposition

        Accepts a spaCy Doc or Span (e.g. a sentence-aligned chunk span).
        """
        relations: list[KGRelationship] = []

        for sent in doc.sents:
            sent_ents = [
                ent for ent in doc.ents
                if ent.start >= sent.start
                and ent.end <= sent.end
                and ent.text.strip().lower() in entity_names   # normalized lookup
            ]
            if len(sent_ents) < 2:
                continue

            for token in sent:
                if token.pos_ not in ("VERB", "AUX"):
                    continue

                subjects = [
                    c for c in token.children
                    if c.dep_ in ("nsubj", "nsubjpass", "csubj", "nsubj:pass")
                ]
                objects = [
                    c for c in token.children
                    if c.dep_ in ("dobj", "obj", "iobj", "attr", "pobj")
                ]

                for subj_tok in subjects:
                    for obj_tok in objects:
                        src = self._find_entity_for_token(subj_tok, sent_ents)
                        tgt = self._find_entity_for_token(obj_tok, sent_ents)
                        if src and tgt and src != tgt:
                            desc, kw = self._make_relation_description(
                                src, tgt, token.lemma_, sent.text
                            )
                            relations.append(KGRelationship(
                                src_id=src,
                                tgt_id=tgt,
                                description=desc,
                                keywords=kw,
                                weight=self.dep_relation_weight,
                                source_id=source_id,
                                file_path=file_path,
                            ))
        return relations

    # ── Co-occurrence fallback ────────────────

    def _extract_cooccurrence_relations(
        self,
        doc,
        entity_names: set[str],
        source_id: str,
        file_path: str,
    ) -> list[KGRelationship]:
        """
        Any two distinct entities within coref_window sentences of each
        other get a weak co-occurrence edge (weight = cooc_relation_weight).
        Useful for catching relationships that never appear in clean SVO form.

        Accepts a spaCy Doc or Span (e.g. a sentence-aligned chunk span).
        """
        relations: list[KGRelationship] = []
        sentences = list(doc.sents)

        for i in range(len(sentences)):
            window = sentences[i: i + self.coref_window]
            window_text = " ".join(s.text for s in window)

            seen: dict[str, None] = {}
            for s in window:
                for ent in doc.ents:
                    if (
                        ent.start >= s.start
                        and ent.end <= s.end
                        and ent.text.strip().lower() in entity_names   # normalized lookup
                    ):
                        seen[ent.text.strip().lower()] = None          # normalized key
            ents_in_window = list(seen.keys())

            for j in range(len(ents_in_window)):
                for k in range(j + 1, len(ents_in_window)):
                    src, tgt = ents_in_window[j], ents_in_window[k]
                    desc, kw = self._make_relation_description(
                        src, tgt, "co-occurrence", window_text
                    )
                    relations.append(KGRelationship(
                        src_id=src,
                        tgt_id=tgt,
                        description=desc,
                        keywords=kw,
                        weight=self.cooc_relation_weight,
                        source_id=source_id,
                        file_path=file_path,
                    ))
        return relations

    # ── Relation deduplication & merging ──────

    @staticmethod
    def _merge_relations(
        relations: list[KGRelationship],
    ) -> list[KGRelationship]:
        """
        Merge all edges between the same entity pair:
          - weight is accumulated
          - keep the longest description
          - deduplicate keywords
          - treat graph as undirected: (A,B) == (B,A)
        """
        merged: dict[tuple[str, str], KGRelationship] = {}
        for rel in relations:
            key = tuple(sorted([rel.src_id, rel.tgt_id]))
            if key not in merged:
                merged[key] = rel
            else:
                existing = merged[key]
                existing.weight += rel.weight
                if len(rel.description) > len(existing.description):
                    existing.description = rel.description
                kw_set = (
                    set(existing.keywords.split(","))
                    | set(rel.keywords.split(","))
                )
                existing.keywords = ",".join(
                    k.strip() for k in kw_set if k.strip()
                )
        return list(merged.values())

    # ── Public API ────────────────────────────

    def build_kg_from_text(
        self,
        text: str,
        file_path: str = "document",
        enable_dep_relations: bool = True,
        enable_cooccurrence: bool = True,
    ) -> CustomKG:
        """
        Build a CustomKG from a single text string.

        Changes vs. original:
          - Parse the full text with spaCy **once**; chunk spans reuse the
            parsed Doc so sentences are never truncated mid-way.
          - Entity names are normalised to ``strip().lower()`` so surface
            variants ("Apple Inc.", "apple inc.") map to the same graph node.

        Args:
            text                : raw input text
            file_path           : label for source tracking in LightRAG
            enable_dep_relations: use SVO dependency extraction
            enable_cooccurrence : use sentence-window co-occurrence fallback
        """
        source_id = hashlib.md5(text.encode()).hexdigest()[:8]

        # ── Pre-split long texts to stay within nlp.max_length ───────────
        # spaCy defaults to 1 000 000 chars; very long documents are split
        # at paragraph/sentence boundaries and each segment is parsed separately.
        segments = self._presplit_text(text)

        all_chunks: list[KGChunk] = []
        entity_registry: dict[str, dict] = {}
        all_relations: list[KGRelationship] = []

        for segment in segments:
            doc = self.nlp(segment)
            seg_chunks, seg_spans, seg_source_ids = self._split_into_sentence_chunks(doc, file_path)
            all_chunks.extend(seg_chunks)

            for _chunk, (tok_start, tok_end), chunk_source_id in zip(seg_chunks, seg_spans, seg_source_ids):
                span = doc[tok_start:tok_end]   # Span preserves all NLP annotations

                # ── NER with normalisation ────────────────────────────────
                for ent in span.ents:
                    etype = self.entity_type_map.get(ent.label_)
                    if etype is None:
                        continue
                    norm_name = ent.text.strip().lower()   # normalised key
                    if not norm_name:
                        continue
                    if norm_name not in entity_registry:
                        entity_registry[norm_name] = {
                            "type": etype,
                            "contexts": [],
                            "freq": 0,
                            "source_id": chunk_source_id,
                            "file_path": file_path,
                        }
                    entity_registry[norm_name]["freq"] += 1
                    for sent in span.sents:
                        if sent.start <= ent.start < sent.end:
                            entity_registry[norm_name]["contexts"].append(sent.text)
                            break

                # ── Relation extraction ───────────────────────────────────
                entity_names = set(entity_registry.keys())
                if enable_dep_relations:
                    all_relations.extend(
                        self._extract_dep_relations(
                            span, entity_names, chunk_source_id, file_path
                        )
                    )
                if enable_cooccurrence:
                    all_relations.extend(
                        self._extract_cooccurrence_relations(
                            span, entity_names, chunk_source_id, file_path
                        )
                    )

        # ── Filter low-frequency entities ────────────────────────────────
        filtered_entities = {
            name: info
            for name, info in entity_registry.items()
            if info["freq"] >= self.min_entity_freq
        }
        valid_names = set(filtered_entities.keys())
        filtered_relations = [
            r for r in all_relations
            if r.src_id in valid_names and r.tgt_id in valid_names
        ]
        merged_relations = self._merge_relations(filtered_relations)

        # ── Assemble ──────────────────────────────────────────────────────
        kg = CustomKG(chunks=all_chunks)
        for name, info in filtered_entities.items():
            kg.entities.append(KGEntity(
                entity_name=name,
                entity_type=info["type"],
                description=self._make_entity_description(
                    name, info["type"], info["contexts"]
                ),
                source_id=info["source_id"],
                file_path=info["file_path"],
            ))
        kg.relationships = merged_relations
        return kg

    def build_kg_from_files(
        self,
        file_paths: list[str],
        encoding: str = "utf-8",
    ) -> CustomKG:
        """Process multiple files and merge into one CustomKG."""
        merged = CustomKG()
        for fp in file_paths:
            print(f"Processing '{fp}' ...")
            with open(fp, encoding=encoding) as f:
                text = f.read()
            kg = self.build_kg_from_text(text, file_path=fp)
            merged.chunks.extend(kg.chunks)
            merged.entities.extend(kg.entities)
            merged.relationships.extend(kg.relationships)
            print(f"  -> {len(kg.entities)} entities, "
                  f"{len(kg.relationships)} relations")
        return merged


# ─────────────────────────────────────────────
# 4. LightRAG integration helper
# ─────────────────────────────────────────────

async def build_and_insert(
    texts: list[str],
    rag,
    file_paths: Optional[list[str]] = None,
    model_name: str = "en_core_web_lg",
    enable_dep_relations: bool = True,
    enable_cooccurrence: bool = True,
) -> dict:
    """
    End-to-end: text list -> spaCy KG -> LightRAG.
    No LLM is called during indexing.
    Embedding runs inside LightRAG as usual.
    """
    builder = SpacyKGBuilder(model_name=model_name)
    totals = {"total_entities": 0, "total_relationships": 0, "total_chunks": 0}

    for i, text in enumerate(texts):
        label = (file_paths[i] if file_paths and i < len(file_paths)
                 else f"doc_{i}")
        print(f"\n[{i+1}/{len(texts)}] '{label}' (len={len(text)}) ...")

        kg = builder.build_kg_from_text(
            text,
            file_path=label,
            enable_dep_relations=enable_dep_relations,
            enable_cooccurrence=enable_cooccurrence,
        )
        print(f"  spaCy: {len(kg.entities)} entities, "
              f"{len(kg.relationships)} relations, {len(kg.chunks)} chunks")

        await rag.ainsert_custom_kg(kg.to_lightrag_format())
        print("  Inserted into LightRAG.")

        totals["total_entities"]      += len(kg.entities)
        totals["total_relationships"] += len(kg.relationships)
        totals["total_chunks"]        += len(kg.chunks)

    return totals


# ─────────────────────────────────────────────
# 5. Demo
# ─────────────────────────────────────────────

DEMO_TEXT = """\
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976
in Cupertino, California. Steve Jobs served as CEO until his resignation in 1985,
and returned as interim CEO in 1997 after Apple acquired NeXT.
Tim Cook succeeded Jobs as CEO in 2011 following Jobs's death.
Apple's flagship products include the iPhone, the Mac, and the iPad.
Microsoft, headquartered in Redmond, Washington, is one of Apple's main rivals
in the personal computer and cloud-computing markets.
Satya Nadella has been Microsoft's CEO since 2014.
"""


async def demo_standalone() -> None:
    """
    Run the spaCy pipeline only — no LightRAG, no API key needed.
    Run this first to check NER and relation quality on your own text.
    """
    builder = SpacyKGBuilder(
        model_name="en_core_web_lg",
        chunk_size=600,
        min_entity_freq=1,
    )
    kg = builder.build_kg_from_text(DEMO_TEXT, file_path="demo.txt")
    kg.print_summary()

    print("\n-- LightRAG format preview (first 1200 chars) --")
    preview = json.dumps(kg.to_lightrag_format(), indent=2, ensure_ascii=False)
    print(preview[:1200], "\n...")


async def demo_with_lightrag() -> None:
    """
    Full integration demo.
    Requires:
        export OPENAI_API_KEY="sk-..."
    Or replace openai_embed / gpt_4o_mini_complete with another provider.
    """
    import os
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    WORKING_DIR = "./rag_storage_spacy"
    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        force_llm_summary_on_merge=999999,   # no LLM at index time
    )
    await rag.initialize_storages()

    stats = await build_and_insert(
        texts=[DEMO_TEXT],
        rag=rag,
        file_paths=["demo.txt"],
    )
    print(f"\nInsert complete: {stats}")

    result = await rag.aquery(
        "Who founded Apple and what products does the company make?",
        param=QueryParam(mode="local"),
    )
    print(f"\nQuery result:\n{result}")

    await rag.finalize_storages()


if __name__ == "__main__":
    # Step 1 — verify spaCy output, no API key needed
    asyncio.run(demo_standalone())

    # Step 2 — uncomment when ready to connect LightRAG
    # asyncio.run(demo_with_lightrag())