import asyncio

import aiohttp
from loguru import logger

from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
    AgentsUnauthorizedException,
)
from src.agents.common.exceptions.token_exceptions import TokenExpiredError

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
        """Function handles response and maps unexpected statuses to exceptions.

        Args:
            response (aiohttp.ClientResponse): Response object
        Returns:
            list | dict: Parsed response data on success, or the ``_RETRY`` sentinel
                when the failure is transient and the request should be retried.
        Raises:
            TokenExpiredError: On 401 with an expired token.
            AgentsUnauthorizedException: On other 401 responses.
            AgentsNotFound: On 404.
            AgentsInputException: On 400.
            DownstreamServiceError: On any other non-2xx status.
        """

        if response.status in (200, 201):
            return await response.json(content_type="application/json")
        if response.status == 401:
            info = await response.json()
            logger.warning(info)
            # TODO revise to more strict rule
            if info.get("message") == "Token expired.":
                raise TokenExpiredError
            raise AgentsUnauthorizedException(message=info)
        if response.status == 500:
            # "reset by peer" is a transient backend hiccup — retry it. Any other
            # 500 is a real downstream failure and must be surfaced, not looped on.
            if response.content_type == "application/json":
                response_info = await response.json()
                if "reset by peer" in response_info.get("error", ""):
                    return _RETRY
            else:
                response_info = await response.text()
            raise DownstreamServiceError(
                service=self.base_url,
                downstream_status=response.status,
                message=f"Downstream service {self.base_url} returned 500",
                error_input=response_info,
            )
        body = await response.text()
        if response.status == 404:
            raise AgentsNotFound(
                message=f"Resource not found at {self.base_url}",
                error_input=body,
            )
        if response.status == 400:
            raise AgentsInputException(
                message=f"Bad request to {self.base_url}",
                error_input=body,
            )
        raise DownstreamServiceError(
            service=self.base_url,
            downstream_status=response.status,
            message=f"Unexpected status {response.status} from {self.base_url}",
            error_input=body,
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

    @staticmethod
    def _with_auth(headers: dict | None, auth_token: str | None) -> dict | None:
        """
        Attach the authorization header when a token is supplied.
        Args:
            headers (dict | None): Existing headers.
            auth_token (str | None): Authorization token.
        Returns:
            dict | None: Headers with the authorization entry, or the originals.
        """

        if not auth_token:
            return headers
        if headers is None:
            return {"Authorization": f"Bearer {auth_token}"}
        headers.update({"Authorization": auth_token})
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        headers: dict | None,
        params: dict | None,
        session: aiohttp.ClientSession,
        data: dict | None = None,
    ) -> dict | list | None:
        """
        Perform an HTTP request with bounded retries and exponential backoff.

        Retries only transient failures (network errors and "reset by peer" 500s);
        terminal statuses raise immediately via :meth:`_check_response_status`.
        Args:
            method (str): "get" or "post".
            endpoint (str): Endpoint url.
            headers (dict | None): Request headers.
            params (dict | None): Query parameters.
            session (aiohttp.ClientSession): Session to use.
            data (dict | None): JSON body for POST requests.
        Returns:
            dict | list | None: Parsed response data.
        Raises:
            DownstreamServiceError: When all retries are exhausted on a transient
                failure, or on a non-retryable downstream status.
        """

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = await self._check_request_params(params)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if method == "get":
                    request_cm = session.get(url=url, headers=headers, params=params)
                else:
                    request_cm = session.post(
                        url=url, headers=headers, params=params, json=data
                    )
                async with request_cm as response:
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
        raise DownstreamServiceError(
            service=self.base_url,
            downstream_status=None,
            message=(
                f"Downstream service {self.base_url} unavailable after "
                f"{self.max_retries} attempts"
            ),
            error_input=repr(last_exc) if last_exc else "transient 500 response",
        ) from last_exc

    async def get(
        self,
        endpoint: str,
        auth_token: str | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> dict | list | None:
        """Function to get data from api
        Args:
            endpoint (str): Endpoint url
            auth_token (str | None): Authorization token.defaults to None
            headers (dict | None): Headers
            params (dict | None): Query parameters
            session (aiohttp.ClientSession | None): Session to use
        Returns:
            dict | list | None: Response data as python object
        """

        headers = self._with_auth(headers, auth_token)
        if not session:
            async with aiohttp.ClientSession() as session:
                return await self._request("get", endpoint, headers, params, session)
        return await self._request("get", endpoint, headers, params, session)

    async def post(
        self,
        endpoint: str,
        auth_token: str | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        data: dict | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> dict | list | None:
        """Function to post data to api.
        Args:
            endpoint (str): Endpoint url.
            auth_token (str | None): Authorization token.defaults to None.
            headers (dict | None): Headers.
            params (dict | None): Query parameters.
            data (dict | None): Data to post to API.
            session (aiohttp.ClientSession | None): Session to use.
        Returns:
            dict | list | None: Response data as python object.
        """

        headers = self._with_auth(headers, auth_token)
        if not session:
            async with aiohttp.ClientSession() as session:
                return await self._request(
                    "post", endpoint, headers, params, session, data=data
                )
        return await self._request(
            "post", endpoint, headers, params, session, data=data
        )
