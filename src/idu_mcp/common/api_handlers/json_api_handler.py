import asyncio

import aiohttp
from fastmcp.exceptions import ToolError
from loguru import logger

# Sentinel returned by _check_response_status to signal a transient failure that
# should be retried rather than returned or raised.
_RETRY = object()

# Transport-level failures worth retrying: transient connectivity issues and the
# "connection reset by peer" / SSL teardown observed under load.
_RETRYABLE_ERRORS = (
    aiohttp.ClientConnectionError,
    aiohttp.ServerDisconnectedError,
    asyncio.TimeoutError,
    ConnectionResetError,
)


class JsonApiHandler:
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        """Initialisation function

        Args:
            base_url (str): Base api url
            max_retries (int): Total attempts for a transient failure before giving
                up. Defaults to 3.
            backoff_base (float): Base delay (seconds) for exponential backoff
                between retries. Defaults to 0.5.
        Returns:
            None
        """

        self.base_url = base_url.rstrip("/")
        self.__name__ = f"{self.base_url}_JSON_API_HANDLER"
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    async def _check_response_status(
        self,
        response: aiohttp.ClientResponse,
    ) -> list | dict | object:
        """Function handles response and maps unexpected statuses to ToolError.

        Args:
            response (aiohttp.ClientResponse): Response object
        Returns:
            list | dict: Parsed response data on success, or the ``_RETRY`` sentinel
                when the failure is transient and the request should be retried.
        Raises:
            ToolError: On any non-2xx, non-transient status, with an actionable
                message for the calling LLM agent.
        """

        if response.status in (200, 201):
            return await response.json(content_type="application/json")
        if response.status == 500:
            # "reset by peer" is a transient backend hiccup — retry it. Any other
            # 500 is a real downstream failure and must be surfaced, not looped on.
            if response.content_type == "application/json":
                response_info = await response.json()
                if "reset by peer" in response_info.get("error", ""):
                    return _RETRY
            else:
                response_info = await response.text()
            raise ToolError(
                f"Urban API ({self.base_url}) returned 500: {response_info}"
            )
        body = await response.text()
        raise ToolError(
            f"Urban API ({self.base_url}) returned status {response.status}: {body}"
        )

    @staticmethod
    async def _check_request_params(
        params: dict[str, str | int | float | bool] | None,
    ) -> dict | None:
        """
        Function checks request parameters
        Args:
            params (dict[str, str | int | float | bool]  | None): Request parameters
        Returns:
            dict | None: Returns modified parameters if they are not empty, otherwise returns None
        """

        if params:
            for key, param in params.items():
                if isinstance(param, bool):
                    params[key] = str(param).lower()
        return params

    async def get(
        self,
        endpoint: str,
        auth_token: str | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> dict | list:
        """Function to get data from api
        Args:
            endpoint (str): Endpoint url
            auth_token (str | None): Authorization token.defaults to None
            headers (dict | None): Headers
            params (dict | None): Query parameters
            session (aiohttp.ClientSession | None): Session to use
        Returns:
            dict | list: Response data as python object
        Raises:
            ToolError: On a non-retryable downstream status or when retries are
                exhausted on a transient failure.
        """

        if auth_token:
            if headers is None:
                headers = {"Authorization": f"Bearer {auth_token}"}
            else:
                headers.update({"Authorization": auth_token})
        if not session:
            async with aiohttp.ClientSession() as session:
                return await self._request(endpoint, headers, params, session)
        return await self._request(endpoint, headers, params, session)

    async def _request(
        self,
        endpoint: str,
        headers: dict | None,
        params: dict | None,
        session: aiohttp.ClientSession,
    ) -> dict | list:
        """
        Perform a GET request with bounded retries and exponential backoff.

        Retries only transient failures (network errors and "reset by peer" 500s);
        terminal statuses raise a ToolError immediately.
        Args:
            endpoint (str): Endpoint url.
            headers (dict | None): Request headers.
            params (dict | None): Query parameters.
            session (aiohttp.ClientSession): Session to use.
        Returns:
            dict | list: Parsed response data.
        Raises:
            ToolError: On a non-retryable status or after exhausting retries.
        """

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = await self._check_request_params(params)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with session.get(
                    url=url, headers=headers, params=params
                ) as response:
                    result = await self._check_response_status(response)
                if result is not _RETRY:
                    return result
                logger.warning(
                    f"Transient failure from {url} "
                    f"(attempt {attempt}/{self.max_retries}), retrying"
                )
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    f"Network error calling {url} "
                    f"(attempt {attempt}/{self.max_retries}): {exc!r}"
                )
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_base * 2 ** (attempt - 1))
        raise ToolError(
            f"Urban API ({self.base_url}) unavailable after "
            f"{self.max_retries} attempts: {last_exc!r}"
        )
