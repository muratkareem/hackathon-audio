import httpx
from pydantic import ValidationError


class InferenceError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, settings):
        self.settings = settings

    def models(self):
        try:
            with httpx.Client(timeout=3, trust_env=False) as client:
                response = client.get(self.settings.ollama_url + "/api/tags")
                response.raise_for_status()
                return [m["name"] for m in response.json()["models"]]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise InferenceError("ollama_unavailable") from exc

    def generate(self, schema, system, content):
        if self.settings.llm_model not in self.models():
            raise InferenceError("ollama_model_missing")
        # Retry malformed output once; transport errors are surfaced immediately.
        for attempt in range(2):
            try:
                with httpx.Client(
                    timeout=self.settings.llm_timeout, trust_env=False
                ) as client:
                    response = client.post(
                        self.settings.ollama_url + "/api/chat",
                        json={
                            "model": self.settings.llm_model,
                            "stream": False,
                            "format": schema.model_json_schema(),
                            "keep_alive": 0,
                            "options": {
                                "temperature": 0,
                                "num_ctx": 8192,
                                "num_predict": 3000,
                            },
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": content},
                            ],
                        },
                    )
                    response.raise_for_status()
                    return schema.model_validate_json(
                        response.json()["message"]["content"]
                    )
            except (ValidationError, ValueError, KeyError):
                if attempt:
                    raise InferenceError("invalid_model_output")
            except httpx.HTTPError as exc:
                raise InferenceError("ollama_request_failed") from exc
