import re
import math
from collections import Counter
import ollama


class MeetingRAG:
    """
    Простой RAG (Retrieval-Augmented Generation) для чата по содержимому встречи.
    
    Retrieval: BM25-подобное ранжирование сегментов по ключевым словам.
    Generation: Ollama с контекстом из top-K релевантных сегментов.
    """

    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.segments = []          # raw_data сегменты
        self.transcript = ""        # полный текст
        self.ai_summary = ""        # AI протокол
        self.chat_history = []      # история чата для контекста

        self.system_prompt = """Ты — ИИ-ассистент по содержимому рабочей встречи. 
Тебе дан транскрипт встречи (расшифровка по спикерам с таймкодами) и AI-протокол.

ПРАВИЛА:
- Отвечай ТОЛЬКО на основе предоставленного контекста. Не выдумывай факты.
- Если информации нет в контексте — честно скажи, что в записи это не обсуждалось.
- Указывай спикера и примерное время, когда упоминалось то, о чём спрашивают.
- Отвечай кратко и по делу, на русском языке.
- Можешь цитировать фрагменты транскрипта для подтверждения.
"""

    def load(self, result: dict):
        """Загружает результат обработки встречи."""
        self.segments = result.get("raw_data", [])
        self.transcript = result.get("formatted_transcript", "")
        self.ai_summary = result.get("ai_summary", "")
        self.chat_history = []

    @property
    def is_loaded(self) -> bool:
        return len(self.segments) > 0

    # ─── Retrieval (BM25-like) ───

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Простая токенизация: нижний регистр, только буквы/цифры, убираем стоп-слова."""
        tokens = re.findall(r'[a-zа-яё0-9]+', text.lower())
        # Минимальный стоп-лист (русский + английский)
        stop = {
            'и', 'в', 'на', 'с', 'по', 'не', 'что', 'это', 'как', 'а', 'но', 'да',
            'он', 'она', 'они', 'мы', 'вы', 'я', 'то', 'все', 'так', 'уже', 'за',
            'от', 'до', 'ну', 'же', 'бы', 'ли', 'к', 'у', 'о', 'из', 'для', 'при',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and',
            'in', 'that', 'it', 'for', 'on', 'with', 'as', 'at', 'by', 'this',
        }
        return [t for t in tokens if t not in stop and len(t) > 1]

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str],
                    avg_dl: float, df: dict[str, int], n_docs: int,
                    k1: float = 1.5, b: float = 0.75) -> float:
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
            # Если токенов нет (например, вопрос из одних стоп-слов), вернём всё
            return self.segments[:top_k]

        # Токенизируем все документы
        docs = []
        for seg in self.segments:
            tokens = self._tokenize(seg.get("text", ""))
            # Добавляем спикера как токен для поиска по имени
            speaker_tokens = self._tokenize(seg.get("speaker", ""))
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

        # Если BM25 ничего не нашёл (все score=0), возвращаем первые top_k
        if not results:
            return self.segments[:top_k]

        return results

    # ─── Generation ───

    def _build_context(self, relevant_segments: list[dict]) -> str:
        """Формирует контекст из релевантных сегментов."""
        lines = []
        for seg in relevant_segments:
            speaker = seg.get("speaker", "?")
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            lang = seg.get("language", "")
            lines.append(f"[{start:.1f}s - {end:.1f}s] {speaker} ({lang}): {text}")
        return "\n".join(lines)

    def ask(self, question: str) -> str:
        """
        Основной метод RAG: retrieval → context building → generation.
        """
        if not self.is_loaded:
            return "❌ Сначала загрузите и обработайте аудиофайл встречи."

        # 1. Retrieval
        relevant = self.retrieve(question, top_k=8)
        context = self._build_context(relevant)

        # 2. Build prompt with context
        user_message = f"""КОНТЕКСТ ВСТРЕЧИ:

--- AI-Протокол ---
{self.ai_summary}

--- Релевантные фрагменты транскрипта ({len(relevant)} из {len(self.segments)}) ---
{context}

--- Вопрос пользователя ---
{question}"""

        # 3. Build messages with chat history (keep last 6 messages for context)
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.chat_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        # 4. Generate
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"num_ctx": 8192}
            )
            answer = response["message"]["content"]
        except Exception as e:
            return f"❌ Ошибка Ollama: {str(e)}"

        # 5. Save to history (store the short question, not the full context)
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer
