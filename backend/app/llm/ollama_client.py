import asyncio

import httpx

from app.config.logging_config import app_logger
from app.config.settings import settings
from app.exceptions.custom_exceptions import (
    LLMConnectionError,
    LLMResponseError,
)


class OllamaClient:

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    async def generate(self,  messages: list) -> str:

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        max_retries = settings.MAX_RETRIES
        timeout = settings.REQUEST_TIMEOUT

        for attempt in range(max_retries):

            try:
                app_logger.info(f"Attempt {attempt + 1} to communicate with Ollama.")
            
                async with httpx.AsyncClient(timeout=timeout) as client:
                
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
        
                    response.raise_for_status()
        
                    data = response.json()
                    app_logger.info("Successfully received response from Ollama.")
        
                    return data["message"]["content"]
        
            except httpx.ConnectError as e:
            
                app_logger.warning(f"Attempt {attempt + 1} failed: {e}.")
    
                if attempt == max_retries - 1:
                    app_logger.error("Maximum retry attempts exceeded.")
                
                    raise LLMConnectionError(
                        "Unable to connect to Ollama."
                    ) from e
                
                wait_time = 2 ** attempt

                app_logger.info(f"Waiting {wait_time} seconds before retrying...")

                await asyncio.sleep(wait_time)
    
            except httpx.HTTPStatusError as e:
            
                app_logger.error(f"Ollama returned HTTP {e.response.status_code}")
    
                raise LLMResponseError(
                    "Ollama returned an invalid response."
                ) from e
    
            except Exception as e:
            
                app_logger.exception("Unexpected error while calling Ollama.")
    
                raise LLMResponseError(
                    "Unexpected LLM error."
                ) from e


ollama_client = OllamaClient()