"""Wrapper around OpenAI embedding models."""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from pydantic import BaseModel, Extra, root_validator
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search docs.

        :param texts: The list of texts to embed.
        :type texts: List[str]

        :return: List of embeddings, one for each text.
        :rtype: List[List[float]]
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text.

        :param text: The text to embed.
        :type text: str

        :return: Embedding for the text.
        :rtype: List[float]
        """

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed search docs asynchronously.

        :param texts: The list of texts to embed.
        :type texts: List[str]

        :return: List of embeddings, one for each text.
        :rtype: List[List[float]]
        """
        raise NotImplementedError

    async def aembed_query(self, text: str) -> List[float]:
        """
        Embed query text asynchronously.

        :param text: The text to embed.
        :type text: str

        :return: Embedding for the text.
        :rtype: List[float]
        """
        raise NotImplementedError


def _create_retry_decorator(embeddings: OpenAIEmbeddings) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _async_retry_decorator(embeddings: OpenAIEmbeddings) -> Any:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    async_retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    def wrap(func: Callable) -> Callable:
        async def wrapped_f(*args: Any, **kwargs: Any) -> Callable:
            async for _ in async_retrying:
                return await func(*args, **kwargs)
            raise AssertionError("this is unreachable")

        return wrapped_f

    return wrap


def embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        return embeddings.client.create(**kwargs)

    return _embed_with_retry(**kwargs)


async def async_embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""

    @_async_retry_decorator(embeddings)
    async def _async_embed_with_retry(**kwargs: Any) -> Any:
        return await embeddings.client.acreate(**kwargs)

    return await _async_embed_with_retry(**kwargs)


class OpenAIEmbeddings(BaseModel, Embeddings):
    """Wrapper around OpenAI embedding models."""

    client: Any  #: :meta private:
    model: str = "text-embedding-ada-002"
    deployment: str = model  # to support Azure OpenAI Service custom deployment names
    openai_api_version: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout in seconds for the OpenAPI request."""
    headers: Any = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        if values["openai_api_type"] in ("azure", "azure_ad", "azuread"):
            default_api_version = "2022-12-01"
        else:
            default_api_version = ""
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        values["openai_organization"] = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

            values["client"] = openai.Embedding
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    @property
    def _invocation_params(self) -> Dict:
        openai_args = {
            "engine": self.deployment,
            "request_timeout": self.request_timeout,
            "headers": self.headers,
            "api_key": self.openai_api_key,
            "organization": self.openai_organization,
            "api_base": self.openai_api_base,
            "api_type": self.openai_api_type,
            "api_version": self.openai_api_version,
        }
        if self.openai_proxy:
            import openai

            openai.proxy = {
                "http": self.openai_proxy,
                "https": self.openai_proxy,
            }  # type: ignore[assignment]  # noqa: E501
        return openai_args

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        encoding = tiktoken.model.encoding_for_model(self.model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens += [token[j: j + self.embedding_ctx_length]]
                indices += [i]

        batched_embeddings = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = embed_with_retry(
                self,
                input=tokens[i: i + _chunk_size],
                **self._invocation_params,
            )
            batched_embeddings += [r["embedding"] for r in response["data"]]

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = embed_with_retry(
                    self,
                    input="",
                    **self._invocation_params,
                )[
                    "data"
                ][0]["embedding"]
            else:
                average = np.average(
                    _result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        encoding = tiktoken.model.encoding_for_model(self.model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens += [token[j: j + self.embedding_ctx_length]]
                indices += [i]

        batched_embeddings = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = await async_embed_with_retry(
                self,
                input=tokens[i: i + _chunk_size],
                **self._invocation_params,
            )
            batched_embeddings += [r["embedding"] for r in response["data"]]

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = (
                    await async_embed_with_retry(
                        self,
                        input="",
                        **self._invocation_params,
                    )
                )["data"][0]["embedding"]
            else:
                average = np.average(
                    _result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # handle large input text
        if len(text) > self.embedding_ctx_length:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            return embed_with_retry(
                self,
                input=[text],
                **self._invocation_params,
            )[
                "data"
            ][0]["embedding"]

    async def _aembedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # handle large input text
        if len(text) > self.embedding_ctx_length:
            return (await self._aget_len_safe_embeddings([text], engine=engine))[0]
        else:
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            return (
                await async_embed_with_retry(
                    self,
                    input=[text],
                    **self._invocation_params,
                )
            )["data"][0]["embedding"]

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """
        Embed search docs.

        :param texts: The list of texts to embed.
        :type texts: List[str]
        :param chunk_size: The chunk size of embeddings. If None, will use the chunk size
            specified by the class.
        :type chunk_size: Optional[int], optional

        :return: List of embeddings, one for each text.
        :rtype: List[List[float]]
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        return self._get_len_safe_embeddings(texts, engine=self.deployment)

    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """
        Embed search docs asynchronously.

        :param texts: The list of texts to embed.
        :type texts: List[str]
        :param chunk_size: The chunk size of embeddings. If None, will use the chunk size
            specified by the class.
        :type chunk_size: Optional[int], optional

        :return: List of embeddings, one for each text.
        :rtype: List[List[float]]
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        return await self._aget_len_safe_embeddings(texts, engine=self.deployment)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text.

        :param text: The text to embed.
        :type text: str

        :return: Embedding for the text.
        :rtype: List[float]
        """
        embedding = self._embedding_func(text, engine=self.deployment)
        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        """
        Embed query text asynchronously.

        :param text: The text to embed.
        :type text: str

        :return: Embedding for the text.
        :rtype: List[float]
        """
        embedding = await self._aembedding_func(text, engine=self.deployment)
        return embedding