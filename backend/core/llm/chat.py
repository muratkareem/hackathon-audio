import re
import math
from collections import Counter
import json
from core.llm.client import LocalLLM, InferenceError
from core.llm.prompts import QA, output_language
from core.schemas import Answer


class MeetingRAG:
    """
    Простой RAG (Retrieval-Augmented Generation) для чата по содержимому встречи.

    Retrieval: BM25-подобное ранжирование сегментов по ключевым словам.
    Generation: Ollama с контекстом из top-K релевантных сегментов.
    """

    def __init__(self):
        self.segments = []

    def load(self, result: dict):
        self.segments = result.get("raw_data", [])

    @property
    def is_loaded(self) -> bool:
        return len(self.segments) > 0

    # ─── Retrieval (BM25-like) ───

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Простая токенизация: нижний регистр, только буквы/цифры, убираем стоп-слова."""
        tokens = re.findall(r"[a-zа-яё0-9]+", text.lower())
        # Минимальный стоп-лист (русский + английский)
        stop = {
            "и",
            "в",
            "на",
            "с",
            "по",
            "не",
            "что",
            "это",
            "как",
            "а",
            "но",
            "да",
            "он",
            "она",
            "они",
            "мы",
            "вы",
            "я",
            "то",
            "все",
            "так",
            "уже",
            "за",
            "от",
            "до",
            "ну",
            "же",
            "бы",
            "ли",
            "к",
            "у",
            "о",
            "из",
            "для",
            "при",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "and",
            "in",
            "that",
            "it",
            "for",
            "on",
            "with",
            "as",
            "at",
            "by",
            "this",
        }
        return [t for t in tokens if t not in stop and len(t) > 1]

    def _bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        avg_dl: float,
        df: dict[str, int],
        n_docs: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """BM25 score для одного документа."""
        dl = len(doc_tokens)
        doc_tf = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt not in doc_tf:
                continue
            tf = doc_tf[qt]
            doc_freq = df.get(qt, 0)
            # IDF component
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            # TF component with length normalization
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_dl)))
            score += idf * tf_norm
        return score

    def retrieve(self, question: str, top_k: int = 8) -> list[dict]:
        """Находит top_k самых релевантных сегментов по BM25."""
        if not self.segments:
            return []

        query_tokens = self._tokenize(question)
        if not query_tokens:
            # Stop-word-only queries provide no retrieval evidence.
            return []

        # Токенизируем все документы
        docs = []
        for seg in self.segments:
            tokens = self._tokenize(seg.get("text", ""))
            # Добавляем спикера как токен для поиска по имени
            speaker_tokens = self._tokenize((seg.get("speaker") or ""))
            docs.append(tokens + speaker_tokens)

        n_docs = len(docs)
        avg_dl = sum(len(d) for d in docs) / max(n_docs, 1)

        # Document frequency
        df = Counter()
        for doc in docs:
            for token in set(doc):
                df[token] += 1

        # Score each segment
        scored = []
        for i, doc_tokens in enumerate(docs):
            score = self._bm25_score(query_tokens, doc_tokens, avg_dl, df, n_docs)
            scored.append((score, i))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scored[:top_k]:
            if score > 0:
                results.append(self.segments[idx])

        # No lexical match: abstain instead of supplying unrelated segments.
        if not results:
            return []

        return results

    # ─── Generation ───

    def ask(self, question, settings, language="en"):
        relevant = self.retrieve(question, top_k=8)
        fallback = (
            "The transcript does not contain enough information."
            if language == "en"
            else "В расшифровке недостаточно информации."
        )
        if not relevant:
            return {
                "answer": fallback,
                "references": [],
                "insufficient_evidence": True,
                "sources": [],
            }
        # Hard context cap; preserve complete individual segments only.
        context, length = [], 0
        for segment in relevant:
            size = len(json.dumps(segment, ensure_ascii=False))
            if length + size <= 12000:
                context.append(segment)
                length += size
        answer = LocalLLM(settings).generate(
            Answer,
            QA + output_language(language),
            json.dumps(
                {"language": language, "question": question, "segments": context},
                ensure_ascii=False,
            ),
        )
        allowed = {s["id"] for s in context}
        if not set(answer.references).issubset(allowed):
            raise InferenceError("invalid_references")
        if answer.insufficient_evidence or not answer.references:
            return {
                "answer": fallback,
                "references": [],
                "insufficient_evidence": True,
                "sources": [],
            }
        return {
            **answer.model_dump(),
            "sources": [s for s in context if s["id"] in answer.references],
        }
