import ollama

class ProtocolGenerator:
    def __init__(self, model_name="llama3"):
        """
        Инициализация генератора протоколов через локальную Ollama.
        model_name: Имя модели, запущенной в Ollama (например 'llama3', 'qwen', 'mistral')
        """
        self.model_name = model_name
        
        # Системный промпт строго по ТЗ хакатона
        self.system_prompt = """Ты — автономный ИИ-протоколист. Твоя задача — проанализировать транскрипт встречи и составить идеальный структурированный протокол на РУССКОМ языке в формате Markdown.
        
        В ответе ОБЯЗАТЕЛЬНО должны быть следующие заголовки (используй ##):
        ## 1. Executive Summary
        Краткая выжимка встречи из 3-5 ключевых предложений.
        
        ## 2. Принятые решения
        Четкий список того, о чем договорились участники.
        
        ## 3. Темы и тезисы
        Разделение встречи на смысловые блоки.
        
        ## 4. Открытые вопросы
        Темы, не получившие финального решения.
        
        ## 5. Таблица поручений (Action Items)
        ОБЯЗАТЕЛЬНО напиши этот раздел! Даже если поручений нет, напиши "Поручений нет".
        Если поручения есть, это должен быть структурированный список задач с полями:
        * **Ответственный**: [Спикер или Имя]
        * **Суть задачи**: [Описание]
        * **Срок выполнения**: [Если озвучен, иначе "Не указан"]
        * **Приоритет**: [Высокий/Средний/Низкий по контексту]
        
        Отвечай строго по тексту встречи. Не выдумывай факты (без галлюцинаций)."""

    def generate_protocol(self, transcript: str):
        """
        Генерация протокола с streaming для лучшего UX.
        Расширен num_ctx до 8192 для длинных транскриптов.
        """
        print(f"🧠 Отправка размеченного текста ({len(transcript)} символов) в Ollama ({self.model_name})...")
        try:
            # Streaming: собираем ответ по частям, выводим по мере генерации
            response_text = ""
            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Вот транскрипт встречи:\n\n{transcript}\n\nСоставь протокол."}
                ],
                stream=True,
                options={
                    "num_ctx": 8192  # Расширяем контекстное окно для длинных транскриптов
                }
            )
            
            for chunk in stream:
                token = chunk['message']['content']
                response_text += token
                print(token, end="", flush=True)
            
            print()  # Завершающий перевод строки
            return response_text
            
        except Exception as e:
            return f"❌ Ошибка при обращении к Ollama: {str(e)}\nУбедитесь, что приложение Ollama запущено и модель {self.model_name} скачана."

    def generate_diagram(self, protocol_text: str):
        print(f"🎨 Генерация диаграммы Excalidraw (Mermaid)...")
        prompt = """Ты эксперт по визуализации данных. На основе текста протокола создай строгую блок-схему в формате Mermaid (graph TD).
        
        ОЧЕНЬ ВАЖНЫЕ ПРАВИЛА (ИНАЧЕ БУДЕТ СИНТАКСИЧЕСКАЯ ОШИБКА):
        1. ВЕРНИ ТОЛЬКО КОД MERMAID, БЕЗ МАРКДАУН-БЛОКОВ (без ```mermaid и ```), БЕЗ ПОЯСНЕНИЙ.
        2. Первая строка ОБЯЗАТЕЛЬНО должна быть: graph TD
        3. НЕ используй кавычки, круглые скобки (), двоеточия : или запятые внутри текста узлов. 
        4. Используй квадратные скобки [] для текста узлов.
        
        Пример правильного ответа:
        graph TD
          A[Проблема] --> B[Обсуждение]
          B --> C[Решение 1]
          B --> D[Решение 2]"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": protocol_text}
                ],
                options={"num_ctx": 4096}
            )
            mermaid_code = response['message']['content'].strip()
            
            # Если LLM добавил текст до или после маркдауна, вытащим строго код
            import re
            match = re.search(r'```(?:mermaid)?(.*?)```', mermaid_code, re.DOTALL)
            if match:
                mermaid_code = match.group(1).strip()
            else:
                # На случай, если LLM не использовал маркдаун вообще
                mermaid_code = mermaid_code.strip()
                
            # Дополнительная очистка от пустых строк, которые иногда ломают Mermaid
            lines = [line.strip() for line in mermaid_code.split('\n') if line.strip()]
            
            # Убедимся, что первая строка точно начинается с graph
            if not lines or not lines[0].startswith("graph"):
                lines = ["graph TD"] + [l for l in lines if "-->" in l]
            
            # Если после фильтрации остался только заголовок, добавим безопасный узел
            if len(lines) == 1:
                lines.append("  Error[Не удалось сгенерировать схему из текста]")
                
            mermaid_code = "\n".join(lines)
            return mermaid_code
        except Exception as e:
            return f"graph TD\n  Error[Ошибка: {str(e)}]"


    def generate_qa_pairs(self, protocol_text: str):
        print(f"❓ Генерация умных вопросов (Q&A)...")
        prompt = '''Ты аналитик. На основе текста протокола придумай 3 важных вопроса, которые могут возникнуть у участников встречи, и сразу дай на них ответы. 
        Верни ответ СТРОГО в формате валидного JSON массива объектов, без маркдауна, без комментариев.
        Формат:
        [
          {"q": "Вопрос 1", "a": "Ответ 1"},
          {"q": "Вопрос 2", "a": "Ответ 2"},
          {"q": "Вопрос 3", "a": "Ответ 3"}
        ]'''
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": protocol_text}
                ],
                options={"num_ctx": 4096}
            )
            import json
            text = response['message']['content'].strip()
            # Clean up potential markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            return json.loads(text.strip())
        except Exception as e:
            print(f"Ошибка Q&A: {e}")
            return []
