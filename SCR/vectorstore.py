"""Interface for vector stores."""
from __future__ import annotations

import asyncio
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
from pydantic import BaseModel, Field, root_validator
from embeddings import Embeddings
from document import Document
from utils import get_prompt_input_key
from base_memory import BaseMemory
from typing import Any, Dict, List, Optional, Union

VST = TypeVar("VST", bound="VectorStore")


class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Run more texts through the embeddings and add to the vector store.

        :param texts: Iterable of strings to add to the vector store.
        :type texts: Iterable[str]

        :param metadatas: Optional list of metadatas associated with the texts.
        :type metadatas: Optional[List[dict]]

        :param kwargs: Vector store specific parameters.
        :type kwargs: Any

        :return: List of IDs from adding the texts into the vector store.
        :rtype: List[str]
        """

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Run more texts through the embeddings and add to the vector store (asynchronous version).

        :param texts: Iterable of strings to add to the vector store.
        :type texts: Iterable[str]

        :param metadatas: Optional list of metadatas associated with the texts.
        :type metadatas: Optional[List[dict]]

        :param kwargs: Vector store specific parameters.
        :type kwargs: Any

        :return: List of IDs from adding the texts into the vector store.
        :rtype: List[str]
        """
        raise NotImplementedError

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        Run more documents through the embeddings and add to the vector store.

        :param documents: Documents to add to the vector store.
        :type documents: List[Document]

        :param kwargs: Vector store specific parameters.
        :type kwargs: Any

        :return: List of IDs of the added texts.
        :rtype: List[str]
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """
        Run more documents through the embeddings and add to the vector store (asynchronous version).

        :param documents: Documents to add to the vector store.
        :type documents: List[Document]

        :param kwargs: Vector store specific parameters.
        :type kwargs: Any

        :return: List of IDs of the added texts.
        :rtype: List[str]
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas, **kwargs)

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        """
        Return documents most similar to the query using the specified search type.

        :param query: The query string.
        :type query: str

        :param search_type: The search type. Valid values are 'similarity' or 'mmr'.
        :type search_type: str

        :param kwargs: Search-specific parameters.
        :type kwargs: Any

        :return: List of documents most similar to the query.
        :rtype: List[Document]
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        elif search_type == "mmr":
            return self.max_marginal_relevance_search(query, **kwargs)
        else:
            raise ValueError(
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity' or 'mmr'."
            )

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        """
        Return documents most similar to the query using the specified search type (asynchronous version).

        :param query: The query string.
        :type query: str

        :param search_type: The search type. Valid values are 'similarity' or 'mmr'.
        :type search_type: str

        :param kwargs: Search-specific parameters.
        :type kwargs: Any

        :return: List of documents most similar to the query.
        :rtype: List[Document]
        """
        if search_type == "similarity":
            return await self.asimilarity_search(query, **kwargs)
        elif search_type == "mmr":
            return await self.amax_marginal_relevance_search(query, **kwargs)
        else:
            raise ValueError(
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity' or 'mmr'."
            )

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Return documents most similar to the query.

        :param query: The query string.
        :type query: str

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents most similar to the query.
        :rtype: List[Document]
        """

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return documents and their relevance scores in the range [0, 1].

        A relevance score of 0 indicates dissimilarity, while a score of 1 indicates maximum similarity.

        :param query: The query string.
        :type query: str

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of tuples of (document, similarity_score).
        :rtype: List[Tuple[Document, float]]
        """
        docs_and_similarities = self._similarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}"
            )

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return docs_and_similarities

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return documents and their relevance scores in the range [0, 1].

        A relevance score of 0 indicates dissimilarity, while a score of 1 indicates maximum similarity.

        :param query: The query string.
        :type query: str

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of tuples of (document, similarity_score).
        :rtype: List[Tuple[Document, float]]
        """
        raise NotImplementedError

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Return documents and their relevance scores in the range [0, 1] (asynchronous version).

        A relevance score of 0 indicates dissimilarity, while a score of 1 indicates maximum similarity.

        :param query: The query string.
        :type query: str

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of tuples of (document, similarity_score).
        :rtype: List[Tuple[Document, float]]
        """

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(
            self.similarity_search_with_relevance_scores, query, k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Return documents most similar to the query (asynchronous version).

        :param query: The query string.
        :type query: str

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents most similar to the query.
        :rtype: List[Document]
        """

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search, query, k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Return documents most similar to the embedding vector.

        :param embedding: The embedding vector to look up documents similar to.
        :type embedding: List[float]

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents most similar to the query vector.
        :rtype: List[Document]
        """
        raise NotImplementedError

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Return documents most similar to the embedding vector (asynchronous version).

        :param embedding: The embedding vector to look up documents similar to.
        :type embedding: List[float]

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents most similar to the query vector.
        :rtype: List[Document]
        """

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_by_vector,
                       embedding, k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to the query and diversity among the selected documents.

        :param query: The query string.
        :type query: str

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param fetch_k: Number of documents to fetch to pass to the MMR algorithm.
        :type fetch_k: int

        :param lambda_mult: Number between 0 and 1 that determines the degree of diversity among the results,
            with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
        :type lambda_mult: float

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents selected by maximal marginal relevance.
        :rtype: List[Document]
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents selected using the maximal marginal relevance (asynchronous version).

        :param query: The query string.
        :type query: str

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param fetch_k: Number of documents to fetch to pass to the MMR algorithm.
        :type fetch_k: int

        :param lambda_mult: Number between 0 and 1 that determines the degree of diversity among the results,
            with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
        :type lambda_mult: float

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents selected by maximal marginal relevance.
        :rtype: List[Document]
        """
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(
            self.max_marginal_relevance_search, query, k, fetch_k, lambda_mult, **kwargs
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to the embedding vector and diversity among the selected documents.

        :param embedding: The embedding vector to look up documents similar to.
        :type embedding: List[float]

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param fetch_k: Number of documents to fetch to pass to the MMR algorithm.
        :type fetch_k: int

        :param lambda_mult: Number between 0 and 1 that determines the degree of diversity among the results,
            with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
        :type lambda_mult: float

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents selected by maximal marginal relevance.
        :rtype: List[Document]
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents selected using the maximal marginal relevance (asynchronous version).

        :param embedding: The embedding vector to look up documents similar to.
        :type embedding: List[float]

        :param k: Number of documents to return. Defaults to 4.
        :type k: int

        :param fetch_k: Number of documents to fetch to pass to the MMR algorithm.
        :type fetch_k: int

        :param lambda_mult: Number between 0 and 1 that determines the degree of diversity among the results,
            with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
        :type lambda_mult: float

        :param kwargs: Additional search-specific parameters.
        :type kwargs: Any

        :return: List of documents selected by maximal marginal relevance.
        :rtype: List[Document]
        """
        raise NotImplementedError

    @classmethod
    def from_documents(
        cls: Type[VST],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VST:
        """
        Return a VectorStore initialized from a list of documents and embeddings.

        :param documents: The list of documents.
        :type documents: List[Document]

        :param embedding: The embeddings object.
        :type embedding: Embeddings

        :param kwargs: Additional parameters specific to the VectorStore implementation.
        :type kwargs: Any

        :return: The initialized VectorStore.
        :rtype: VectorStore
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    async def afrom_documents(
        cls: Type[VST],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VST:
        """
        Return a VectorStore initialized from a list of documents and embeddings (asynchronous version).

        :param documents: The list of documents.
        :type documents: List[Document]

        :param embedding: The embeddings object.
        :type embedding: Embeddings

        :param kwargs: Additional parameters specific to the VectorStore implementation.
        :type kwargs: Any

        :return: The initialized VectorStore.
        :rtype: VectorStore
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return await cls.afrom_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    @abstractmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """
        Return a VectorStore initialized from a list of texts and embeddings.

        :param texts: The list of texts.
        :type texts: List[str]

        :param embedding: The embeddings object.
        :type embedding: Embeddings

        :param metadatas: Optional list of metadatas associated with the texts.
        :type metadatas: Optional[List[dict]]

        :param kwargs: Additional parameters specific to the VectorStore implementation.
        :type kwargs: Any

        :return: The initialized VectorStore.
        :rtype: VectorStore
        """

    @classmethod
    async def afrom_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """
        Return a VectorStore initialized from a list of texts and embeddings (asynchronous version).

        :param texts: The list of texts.
        :type texts: List[str]

        :param embedding: The embeddings object.
        :type embedding: Embeddings

        :param metadatas: Optional list of metadatas associated with the texts.
        :type metadatas: Optional[List[dict]]

        :param kwargs: Additional parameters specific to the VectorStore implementation.
        :type kwargs: Any

        :return: The initialized VectorStore.
        :rtype: VectorStore
        """
        raise NotImplementedError

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """
        Return a VectorStoreRetriever instance for the VectorStore.

        :param kwargs: Additional parameters for the VectorStoreRetriever.
        :type kwargs: Any

        :return: The VectorStoreRetriever instance.
        :rtype: VectorStoreRetriever
        """
        return VectorStoreRetriever(vectorstore=self, **kwargs)

class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant for a query.

        :param query: String to find relevant documents for.
        :type query: str

        :return: List of relevant documents.
        :rtype: List[Document]
        """

    @abstractmethod
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant for a query.

        :param query: String to find relevant documents for.
        :type query: str

        :return: List of relevant documents.
        :rtype: List[Document]
        """

class VectorStoreRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        search_type = values["search_type"]
        if search_type not in cls.allowed_search_types:
            raise ValueError(
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
        if search_type == "similarity_score_threshold":
            score_threshold = values["search_kwargs"].get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                raise ValueError(
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
        return values

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant for a query.

        :param query: String to find relevant documents for.
        :type query: str

        :return: List of relevant documents.
        :rtype: List[Document]
        """
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant for a query.

        :param query: String to find relevant documents for.
        :type query: str

        :return: List of relevant documents.
        :rtype: List[Document]
        """
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """
        Add documents to vectorstore.

        :param documents: List of documents to add.
        :type documents: List[Document]
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any

        :return: List of added document IDs.
        :rtype: List[str]
        """
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """
        Add documents to vectorstore asynchronously.

        :param documents: List of documents to add.
        :type documents: List[Document]
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any

        :return: List of added document IDs.
        :rtype: List[str]
        """
        return await self.vectorstore.aadd_documents(documents, **kwargs)
    
class VectorStoreRetrieverMemory(BaseMemory):
    """Class for a VectorStore-backed memory object."""

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStoreRetriever object to connect to."""

    memory_key: str = "history"  #: :meta private:
    """Key name to locate the memories in the result of load_memory_variables."""

    input_key: Optional[str] = None
    """Key name to index the inputs to load_memory_variables."""

    return_docs: bool = False
    """Whether or not to return the result of querying the database directly."""

    @property
    def memory_variables(self) -> List[str]:
        """The list of keys emitted from the load_memory_variables method."""
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Union[List[Document], str]]:
        """
        Return history buffer.

        :param inputs: Dictionary of input values.
        :type inputs: Dict[str, Any]

        :return: Dictionary containing the memory variables.
        :rtype: Dict[str, Union[List[Document], str]]
        """
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        result: Union[List[Document], str]
        if not self.return_docs:
            result = "\n".join([doc.page_content for doc in docs])
        else:
            result = docs
        return {self.memory_key: result}

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """
        Format context from this conversation to buffer.

        :param inputs: Dictionary of input values.
        :type inputs: Dict[str, Any]
        :param outputs: Dictionary of output values.
        :type outputs: Dict[str, str]

        :return: List of formatted documents.
        :rtype: List[Document]
        """
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {k: v for k,
                           v in inputs.items() if k != self.memory_key}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context from this conversation to buffer.

        :param inputs: Dictionary of input values.
        :type inputs: Dict[str, Any]
        :param outputs: Dictionary of output values.
        :type outputs: Dict[str, str]
        """
        documents = self._form_documents(inputs, outputs)
        self.retriever.add_documents(documents)

    def clear(self) -> None:
        """Nothing to clear."""