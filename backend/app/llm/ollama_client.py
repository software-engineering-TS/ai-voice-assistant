import asyncio
import json
from typing import Any, AsyncGenerator

import httpx

from app.config.logging_config import app_logger
from app.config.settings import settings
from app.exceptions.custom_exceptions import (
    LLMConnectionError,
    LLMResponseError,
)


class OllamaClient:
    """
    Async client for communicating with Ollama.

    Responsibilities:

    - Generate normal AI responses.
    - Stream AI responses.
    - Handle connection failures.
    - Handle timeouts.
    - Retry transient failures.
    - Validate Ollama responses.
    - Manage the shared HTTP client lifecycle.
    """

    # HTTP status codes that may be temporarily unavailable.
    RETRYABLE_STATUS_CODES = {
        429,
    }

    def __init__(self) -> None:

        # --------------------------------------------------------
        # Configuration
        # --------------------------------------------------------

        self.base_url = (
            settings.OLLAMA_BASE_URL
            .strip()
            .rstrip("/")
        )

        self.model = (
            settings.OLLAMA_MODEL
            .strip()
        )

        self.timeout = max(
            1,
            int(settings.REQUEST_TIMEOUT),
        )

        self.max_retries = max(
            1,
            int(settings.MAX_RETRIES),
        )

        # --------------------------------------------------------
        # Validate Configuration
        # --------------------------------------------------------

        if not self.base_url:

            raise ValueError(
                "OLLAMA_BASE_URL cannot be empty."
            )

        if not self.model:

            raise ValueError(
                "OLLAMA_MODEL cannot be empty."
            )

        # --------------------------------------------------------
        # HTTP Client
        # --------------------------------------------------------

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=self.timeout,
                connect=self.timeout,
            ),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "AI-Assistant/1.0",
            },
        )

        app_logger.info(
            "Ollama client initialized | "
            f"Model={self.model} | "
            f"BaseURL={self.base_url} | "
            f"Timeout={self.timeout}s | "
            f"MaxRetries={self.max_retries}"
        )

    # ============================================================
    # Close Client
    # ============================================================

    async def close(self) -> None:
        """
        Close the shared HTTP client.
        """

        if self.client.is_closed:

            return

        try:

            await self.client.aclose()

            app_logger.info(
                "Ollama HTTP client closed."
            )

        except Exception:

            app_logger.exception(
                "Failed to close Ollama HTTP client."
            )

            raise

    # ============================================================
    # Generate Response
    # ============================================================

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a complete response from Ollama.

        Parameters:
            messages:
                Chat messages sent to Ollama.

            system_prompt:
                Optional Ollama system prompt.

        Returns:
            Generated assistant response.
        """

        self._validate_messages(
            messages
        )

        payload = self._build_payload(
            messages=messages,
            stream=False,
            system_prompt=system_prompt,
        )

        endpoint = (
            f"{self.base_url}/api/chat"
        )

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                app_logger.info(
                    "Ollama generation request | "
                    f"Attempt={attempt}/"
                    f"{self.max_retries} | "
                    f"Model={self.model}"
                )

                response = (
                    await self.client.post(
                        endpoint,
                        json=payload,
                    )
                )

                # ------------------------------------------------
                # HTTP Status Handling
                # ------------------------------------------------

                if response.is_error:

                    status_code = (
                        response.status_code
                    )

                    app_logger.warning(
                        "Ollama returned HTTP error | "
                        f"Status={status_code} | "
                        f"Attempt={attempt}"
                    )

                    if (
                        self._is_retryable_status(
                            status_code
                        )
                        and attempt
                        < self.max_retries
                    ):

                        await self._wait_before_retry(
                            attempt
                        )

                        continue

                    error_detail = (
                        self._extract_error_detail(
                            response
                        )
                    )

                    raise LLMResponseError(
                        "Ollama request failed | "
                        f"HTTP {status_code}"
                        + (
                            f" | {error_detail}"
                            if error_detail
                            else ""
                        )
                    )

                # ------------------------------------------------
                # Parse JSON
                # ------------------------------------------------

                data = self._parse_json(
                    response
                )

                # ------------------------------------------------
                # Extract Response
                # ------------------------------------------------

                content = (
                    self._extract_content(
                        data
                    )
                )

                app_logger.info(
                    "Ollama response received successfully | "
                    f"Characters={len(content)}"
                )

                return content

            # ----------------------------------------------------
            # Connection Error
            # ----------------------------------------------------

            except httpx.ConnectError as exc:

                app_logger.warning(
                    "Ollama connection failed | "
                    f"Attempt={attempt}/"
                    f"{self.max_retries}"
                )

                if attempt >= self.max_retries:

                    raise LLMConnectionError(
                        "Unable to connect to Ollama."
                    ) from exc

                await self._wait_before_retry(
                    attempt
                )

            # ----------------------------------------------------
            # Timeout
            # ----------------------------------------------------

            except httpx.TimeoutException as exc:

                app_logger.warning(
                    "Ollama request timed out | "
                    f"Attempt={attempt}/"
                    f"{self.max_retries}"
                )

                if attempt >= self.max_retries:

                    raise LLMConnectionError(
                        "Ollama request timed out."
                    ) from exc

                await self._wait_before_retry(
                    attempt
                )

            # ----------------------------------------------------
            # Network HTTP Error
            # ----------------------------------------------------

            except httpx.HTTPError as exc:

                app_logger.exception(
                    "Ollama HTTP client error."
                )

                raise LLMConnectionError(
                    "Communication with Ollama failed."
                ) from exc

            # ----------------------------------------------------
            # Application LLM Errors
            # ----------------------------------------------------

            except LLMResponseError:

                raise

            except LLMConnectionError:

                raise

            # ----------------------------------------------------
            # Unexpected Error
            # ----------------------------------------------------

            except Exception as exc:

                app_logger.exception(
                    "Unexpected error while calling Ollama."
                )

                raise LLMResponseError(
                    "Unexpected LLM error."
                ) from exc

        raise LLMResponseError(
            "LLM generation failed."
        )

    # ============================================================
    # Streaming Response
    # ============================================================

    async def stream_generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream Ollama response incrementally.

        Streaming requests are intentionally not retried after
        the stream starts because retrying could duplicate tokens
        already delivered to the client.
        """

        self._validate_messages(
            messages
        )

        payload = self._build_payload(
            messages=messages,
            stream=True,
            system_prompt=system_prompt,
        )

        endpoint = (
            f"{self.base_url}/api/chat"
        )

        received_content = False
        completed = False

        try:

            app_logger.info(
                "Starting Ollama streaming request | "
                f"Model={self.model}"
            )

            async with self.client.stream(
                "POST",
                endpoint,
                json=payload,
            ) as response:

                # ------------------------------------------------
                # HTTP Status
                # ------------------------------------------------

                if response.is_error:

                    status_code = (
                        response.status_code
                    )

                    error_detail = (
                        self._extract_error_detail(
                            response
                        )
                    )

                    app_logger.error(
                        "Ollama streaming HTTP error | "
                        f"Status={status_code}"
                    )

                    raise LLMResponseError(
                        "Ollama streaming request failed | "
                        f"HTTP {status_code}"
                        + (
                            f" | {error_detail}"
                            if error_detail
                            else ""
                        )
                    )

                # ------------------------------------------------
                # Process Stream
                # ------------------------------------------------

                async for line in (
                    response.aiter_lines()
                ):

                    if not line:
                        continue

                    try:

                        data = json.loads(
                            line
                        )

                    except json.JSONDecodeError:

                        app_logger.warning(
                            "Ignoring invalid JSON "
                            "from Ollama stream."
                        )

                        continue

                    # ------------------------------------------------
                    # Extract Token
                    # ------------------------------------------------

                    message = data.get(
                        "message",
                        {},
                    )

                    if isinstance(
                        message,
                        dict,
                    ):

                        token = message.get(
                            "content",
                            "",
                        )

                        if (
                            isinstance(
                                token,
                                str,
                            )
                            and token
                        ):

                            received_content = True

                            yield token

                    # ------------------------------------------------
                    # Completion
                    # ------------------------------------------------

                    if data.get(
                        "done"
                    ) is True:

                        completed = True

                        app_logger.info(
                            "Ollama streaming completed."
                        )

                        break

                # ------------------------------------------------
                # Validate Stream
                # ------------------------------------------------

                if not received_content:

                    raise LLMResponseError(
                        "Ollama returned an empty "
                        "streaming response."
                    )

                if not completed:

                    app_logger.warning(
                        "Ollama stream ended without "
                        "a completion event."
                    )

        except httpx.ConnectError as exc:

            app_logger.exception(
                "Ollama streaming connection failed."
            )

            raise LLMConnectionError(
                "Unable to connect to Ollama."
            ) from exc

        except httpx.TimeoutException as exc:

            app_logger.exception(
                "Ollama streaming request timed out."
            )

            raise LLMConnectionError(
                "Ollama streaming request timed out."
            ) from exc

        except httpx.HTTPError as exc:

            app_logger.exception(
                "Ollama streaming HTTP error."
            )

            raise LLMConnectionError(
                "Communication with Ollama failed."
            ) from exc

        except LLMResponseError:

            raise

        except LLMConnectionError:

            raise

        except Exception as exc:

            app_logger.exception(
                "Unexpected Ollama streaming error."
            )

            raise LLMResponseError(
                "Streaming generation failed."
            ) from exc

    # ============================================================
    # Build Payload
    # ============================================================

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        stream: bool,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        """
        Build Ollama API request payload.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        if system_prompt:

            system_prompt = (
                system_prompt.strip()
            )

            if system_prompt:

                payload[
                    "system"
                ] = system_prompt

        return payload

    # ============================================================
    # Validate Messages
    # ============================================================

    @staticmethod
    def _validate_messages(
        messages: list[dict[str, Any]],
    ) -> None:
        """
        Validate Ollama message structure.
        """

        if not messages:

            raise LLMResponseError(
                "No messages were provided to Ollama."
            )

        if not isinstance(
            messages,
            list,
        ):

            raise LLMResponseError(
                "Ollama messages must be a list."
            )

        for index, message in enumerate(
            messages
        ):

            if not isinstance(
                message,
                dict,
            ):

                raise LLMResponseError(
                    f"Invalid message at index {index}."
                )

            role = message.get(
                "role"
            )

            content = message.get(
                "content"
            )

            if not isinstance(
                role,
                str,
            ) or not role.strip():

                raise LLMResponseError(
                    f"Invalid message role "
                    f"at index {index}."
                )

            if not isinstance(
                content,
                str,
            ) or not content.strip():

                raise LLMResponseError(
                    f"Invalid message content "
                    f"at index {index}."
                )

    # ============================================================
    # Parse JSON
    # ============================================================

    @staticmethod
    def _parse_json(
        response: httpx.Response,
    ) -> dict[str, Any]:
        """
        Parse Ollama JSON response.
        """

        try:

            data = response.json()

        except json.JSONDecodeError as exc:

            app_logger.exception(
                "Ollama returned invalid JSON."
            )

            raise LLMResponseError(
                "Ollama returned invalid JSON."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):

            raise LLMResponseError(
                "Ollama returned an invalid response."
            )

        return data

    # ============================================================
    # Extract Content
    # ============================================================

    @staticmethod
    def _extract_content(
        data: dict[str, Any],
    ) -> str:
        """
        Extract assistant content from Ollama response.
        """

        message = data.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):

            raise LLMResponseError(
                "Ollama response does not contain "
                "a valid message."
            )

        content = message.get(
            "content",
            "",
        )

        if not isinstance(
            content,
            str,
        ):

            raise LLMResponseError(
                "Ollama returned invalid message content."
            )

        content = content.strip()

        if not content:

            raise LLMResponseError(
                "Ollama returned an empty response."
            )

        return content

    # ============================================================
    # Extract HTTP Error Detail
    # ============================================================

    @staticmethod
    def _extract_error_detail(
        response: httpx.Response,
    ) -> str:
        """
        Safely extract an error message from Ollama.
        """

        try:

            data = response.json()

            if isinstance(
                data,
                dict,
            ):

                error = data.get(
                    "error"
                )

                if isinstance(
                    error,
                    str,
                ):

                    return error.strip()

        except Exception:

            pass

        try:

            return response.text.strip()[
                :500
            ]

        except Exception:

            return ""

    # ============================================================
    # Retryable Status
    # ============================================================

    @classmethod
    def _is_retryable_status(
        cls,
        status_code: int,
    ) -> bool:
        """
        Determine whether an HTTP status should be retried.

        Retry:
            429
            5xx

        Do not retry:
            Other 4xx responses.
        """

        return (
            status_code >= 500
            or status_code
            in cls.RETRYABLE_STATUS_CODES
        )

    # ============================================================
    # Retry Delay
    # ============================================================

    async def _wait_before_retry(
        self,
        attempt: int,
    ) -> None:
        """
        Wait using exponential backoff.

        Maximum delay: 8 seconds.
        """

        wait_time = min(
            2 ** (
                attempt - 1
            ),
            8,
        )

        app_logger.info(
            "Retrying Ollama request | "
            f"Delay={wait_time}s"
        )

        await asyncio.sleep(
            wait_time
        )


# ============================================================
# Shared Ollama Client
# ============================================================

ollama_client = OllamaClient()