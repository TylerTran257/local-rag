import json
import logging
from time import perf_counter
from typing import AsyncIterator

import httpx
from httpx import NetworkError, TimeoutException

from app.settings import settings

logger = logging.getLogger(__name__)


class GenerationServiceError(Exception):
    pass


class GenerationService:
    def __init__(self) -> None:
        self.base_url = settings.generation_base_url
        self.endpoint = settings.generation_endpoint
        self.timeout = settings.generation_timeout
        self.temperature = settings.generation_temperature
        self.max_output_tokens = settings.generation_max_output_tokens
        self.max_context_chars = settings.generation_max_context_chars
        self.max_chars_per_chunk = settings.generation_max_chars_per_chunk

    def answer_question(self, question: str, sources: list[dict]) -> str:
        started_at = perf_counter()
        if not question.strip():
            raise GenerationServiceError("Question must not be empty")

        if len(sources) == 0:
            raise GenerationServiceError("No sources were provided for generation")

        messages = [
            {"role": "system", "content": self._build_system_message()},
            {"role": "user", "content": self._build_user_message(question, sources)},
        ]
        payload = self._create_payload(messages)
        data = self._post_chat_completion(payload)
        answer = self._parse_response(data)

        logger.info(
            "event=answer_question_completed source_count=%s answer_length=%s duration_ms=%s",
            len(sources),
            len(answer),
            round((perf_counter() - started_at) * 1000, 2),
        )
        return answer

    async def stream_answer_question(self, question: str, sources: list[dict]):
        started_at = perf_counter()
        if not question.strip():
            raise GenerationServiceError("Question must not be empty")

        if len(sources) == 0:
            raise GenerationServiceError("No sources were provided for generation")

        messages = [
            {"role": "system", "content": self._build_system_message()},
            {"role": "user", "content": self._build_user_message(question, sources)},
        ]
        payload = self._create_payload(messages, stream=True)
        async for token in self._stream_chat_completion(payload):
            yield token

        logger.info(
            "event=stream_answer_question_completed source_count=%s duration_ms=%s",
            len(sources),
            round((perf_counter() - started_at) * 1000, 2),
        )

    def _build_system_message(self) -> str:
        """return the fixed grounding rules"""

        return """
            Return the final answer in the assistant response content only.
            Cite factual claims with inline citation using source number in square brackets, like [1] or [2].
            Use only the provided source numbers.
            Do not invent citations.
            If a claim is supported by multiple sources, you may cite multiple sources like [1][2]
            Do not output reasoning.
            Do not output analysis.
            Return only final answer in 2-4 sentences.
            """.strip()

    def _build_user_message(self, question: str, sources: list[dict]) -> str:
        """build the full user-visible prompt body"""

        formatted_context = self._format_sources_for_context(sources)

        return f"""
            Question:
            {question}

            Context:
            {formatted_context}
        """.strip()

    def _format_sources_for_context(self, sources: list[dict]) -> str:
        """
        apply the char budget across all retrieved chunks
        format chunks in retrieval order
        """

        blocks = []
        total_chars = 0

        for index, source in enumerate(sources, start=1):
            raw_text = source.get("text", "")
            truncated_text = self._truncate_source_text(
                raw_text, self.max_chars_per_chunk
            )

            header = (
                f"[Source {index} | "
                f"file={source.get('original_filename', 'unknown')} | "
                f"chunk={source.get('chunk_index', '?')} | "
                f"score={source.get('score', '?')}]"
            )

            block = f"{header}\n{truncated_text}".strip()

            if total_chars + len(block) > self.max_context_chars:
                break

            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks)

    def _truncate_source_text(self, text: str, max_chars: int) -> str:
        """
        cap one chunk so a single source cannot dominate the prompt
        """

        cleaned_text = text.strip()
        if len(cleaned_text) <= max_chars:
            return cleaned_text

        return cleaned_text[:max_chars].rstrip() + "..."

    def _create_payload(
        self, messages: list[dict[str, str]], stream: bool = False
    ) -> dict:
        return {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": stream,
        }

    def _post_chat_completion(self, payload: dict) -> dict:
        """
        make the HTTP request to llama-server
        translate transport/upstream failures into one clean service error

        useful:
        - exception catching: what failures do I want to present to my app?
        except Exception as exc:
            print(type(exc), repr(exc))
            raise
        """
        started_at = perf_counter()
        url = self.base_url + self.endpoint

        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except TimeoutException as exc:
            logger.error(
                "event=post_chat_completion_failed error_message=%s",
                "Generation request timed out",
            )
            raise GenerationServiceError("Generation request timed out") from exc
        except NetworkError as exc:
            logger.error(
                "event=post_chat_completion_failed error_message=%s",
                "Generation service is unavailable",
            )
            raise GenerationServiceError("Generation service is unavailable") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "event=post_chat_completion_failed error_message=%s",
                f"Generation service returned HTTP {exc.response.status_code}",
            )
            raise GenerationServiceError(
                f"Generation service returned HTTP {exc.response.status_code}"
            ) from exc

        try:
            response_json = response.json()
        except ValueError as exc:
            logger.error(
                "event=post_chat_completion_failed error_message=%s",
                "Generation service returned invalid JSON",
            )
            raise GenerationServiceError(
                "Generation service returned invalid JSON"
            ) from exc

        logger.info(
            "event=post_chat_completion_completed duration_ms=%s",
            round((perf_counter() - started_at) * 1000, 2),
        )
        return response_json

    def _parse_response(self, response_json: dict) -> str:
        choices = response_json.get("choices")
        if not choices:
            raise GenerationServiceError("Generation response was malformed")

        first_choice = choices[0]
        message = first_choice.get("message")
        if not message:
            raise GenerationServiceError("Generation response was malformed")

        content = message.get("content")
        if not content or not content.strip():
            raise GenerationServiceError("Generation response was empty")

        return content.strip()

    async def _stream_chat_completion(self, payload: dict) -> AsyncIterator[str]:
        started_at = perf_counter()
        first_token_logged = False
        url = self.base_url + self.endpoint

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        if not line.startswith("data: "):
                            continue

                        data = line.removeprefix("data: ")
                        if data == "[DONE]":
                            break

                        event = json.loads(data)
                        choices = event.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content")

                        if content:
                            if not first_token_logged:
                                logger.info(
                                    "event=time_to_first_token duration_ms=%s",
                                    round((perf_counter() - started_at) * 1000, 2),
                                )
                                first_token_logged = True
                            yield content
        except TimeoutException as exc:
            logger.error(
                "event=stream_chat_completion_failed error_message=%s",
                "Generation request timed out",
            )
            raise GenerationServiceError("Generation request timed out") from exc
        except NetworkError as exc:
            logger.error(
                "event=stream_chat_completion_failed error_message=%s",
                "Generation service is unavailable",
            )
            raise GenerationServiceError("Generation service is unavailable") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "event=stream_chat_completion_failed error_message=%s",
                f"Generation service returned HTTP {exc.response.status_code}",
            )
            raise GenerationServiceError(
                f"Generation service returned HTTP {exc.response.status_code}"
            ) from exc
        except ValueError as exc:
            logger.error(
                "event=stream_chat_completion_failed error_message=%s",
                "Generation stream was malformed",
            )
            raise GenerationServiceError("Generation stream was malformed") from exc
        logger.info(
            "event=stream_chat_completion_completed duration_ms=%s",
            round((perf_counter() - started_at) * 1000, 2),
        )
