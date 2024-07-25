"""Wrapper around ChromaDB embeddings platform."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np

from document import Document
from embeddings import Embeddings
from vectorstore import VectorStore
from utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import chromadb
    import chromadb.config

logger = logging.getLogger(__name__)


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    """Convert results to a list of documents."""
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class Chroma(VectorStore):
    """Wrapper around ChromaDB embeddings platform."""

    def __init__(
        self,
        collection_name: str = "gentopia",
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[chromadb.Client] = None,
    ) -> None:
        """
        Initialize Chroma with the provided settings.

        :param collection_name: Name of the collection, defaults to "gentopia".
        :type collection_name: str, optional
        :param embedding_function: Embedding function, defaults to None.
        :type embedding_function: Optional[Embeddings], optional
        :param persist_directory: Directory to persist the collection, defaults to None.
        :type persist_directory: Optional[str], optional
        :param client_settings: Chroma client settings, defaults to None.
        :type client_settings: Optional[chromadb.config.Settings], optional
        :param collection_metadata: Metadata for the collection, defaults to None.
        :type collection_metadata: Optional[Dict], optional
        :param client: Chroma client object, defaults to None.
        :type client: Optional[chromadb.Client], optional
        """
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )

        if client is not None:
            self._client = client
        else:
            if client_settings:
                self._client_settings = client_settings
            else:
                self._client_settings = chromadb.config.Settings()
                if persist_directory is not None:
                    self._client_settings = chromadb.config.Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=persist_directory,
                    )
            self._client = chromadb.Client(self._client_settings)

        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function.embed_documents
            if self._embedding_function is not None
            else None,
            metadata=collection_metadata,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Run more texts through the embeddings and add them to the vectorstore.

        :param texts: Texts to add to the vectorstore.
        :type texts: Iterable[str]
        :param metadatas: Optional list of metadatas, defaults to None.
        :type metadatas: Optional[List[dict]], optional
        :param ids: Optional list of IDs, defaults to None.
        :type ids: Optional[List[str]], optional
        :return: List of IDs of the added texts.
        :rtype: List[str]
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
        self._collection.add(
            metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Run similarity search with Chroma.

        :param query: Query text to search for.
        :type query: str
        :param k: Number of results to return, defaults to 4.
        :type k: int, optional
        :param filter: Filter by metadata, defaults to None.
        :type filter: Optional[Dict[str, str]], optional
        :return: List of documents most similar to the query text.
        :rtype: List[Document]
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents most similar to the embedding vector.

        :param embedding: Embedding to look up documents similar to.
        :type embedding: List[float]
        :param k: Number of Documents to return, defaults to 4.
        :type k: int, optional
        :param filter: Filter by metadata, defaults to None.
        :type filter: Optional[Dict[str, str]], optional
        :return: List of Documents most similar to the query vector.
        :rtype: List[Document]
        """
        results = self._collection.query(
            query_embeddings=embedding, n_results=k, where=filter
        )
        return _results_to_docs(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Run similarity search with Chroma with distance.

        :param query: Query text to search for.
        :type query: str
        :param k: Number of results to return, defaults to 4.
        :type k: int, optional
        :param filter: Filter by metadata, defaults to None.
        :type filter: Optional[Dict[str, str]], optional
        :return: List of documents most similar to the query text with distance in float.
        :rtype: List[Tuple[Document, float]]
        """
        if self._embedding_function is None:
            results = self._collection.query(
                query_texts=[query], n_results=k, where=filter
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding], n_results=k, where=filter
            )

        return _results_to_docs_and_scores(results)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

        :param embedding: Embedding to look up documents similar to.
        :type embedding: List[float]
        :param k: Number of Documents to return, defaults to 4.
        :type k: int, optional
        :param fetch_k: Number of Documents to fetch to pass to MMR algorithm.
        :type fetch_k: int, optional
        :param filter: Filter by metadata, defaults to None.
        :type filter: Optional[Dict[str, str]], optional
        :return: List of Documents selected by maximal marginal relevance.
        :rtype: List[Document]
        """

        results = self._collection.query(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=filter,
            include=["metadatas", "documents", "distances", "embeddings"],
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32), results["embeddings"][0], k=k
        )

        candidates = _results_to_docs(results)

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return documents selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

        :param query: Text to look up documents similar to.
        :type query: str
        :param k: Number of Documents to return, defaults to 4.
        :type k: int, optional
        :param fetch_k: Number of Documents to fetch to pass to MMR algorithm.
        :type fetch_k: int, optional
        :param filter: Filter by metadata, defaults to None.
        :type filter: Optional[Dict[str, str]], optional
        :return: List of Documents selected by maximal marginal relevance.
        :rtype: List[Document]
        """
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
            )

        embedding = self._embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, filter
        )
        return docs

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)

    def persist(self) -> None:
        """Persist the collection.

        This can be used to explicitly persist the data to disk.
        It will also be called automatically when the object is destroyed.
        """
        if self._persist_directory is None:
            raise ValueError(
                "You must specify a persist_directory on"
                "creation to persist the collection."
            )
        self._client.persist()

    @classmethod
    def from_texts(
        cls: Type[Chroma],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "gentopia",
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,
        **kwargs: Any,
    ) -> Chroma:
        """
        Create a Chroma vectorstore from a list of raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        :param texts: List of texts to add to the collection.
        :type texts: List[str]
        :param embedding: Embedding function, defaults to None.
        :type embedding: Optional[Embeddings], optional
        :param metadatas: List of metadatas, defaults to None.
        :type metadatas: Optional[List[dict]], optional
        :param ids: List of document IDs, defaults to None.
        :type ids: Optional[List[str]], optional
        :param collection_name: Name of the collection to create, defaults to "gentopia".
        :type collection_name: str, optional
        :param persist_directory: Directory to persist the collection, defaults to None.
        :type persist_directory: Optional[str], optional
        :param client_settings: Chroma client settings, defaults to None.
        :type client_settings: Optional[chromadb.config.Settings], optional
        :param client: Chroma client object, defaults to None.
        :type client: Optional[chromadb.Client], optional
        :return: Chroma vectorstore.
        :rtype: Chroma
        """
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
        )
        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "gentopia",
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # Add this line
        **kwargs: Any,
    ) -> Chroma:
        """
        Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        :param collection_name: Name of the collection to create.
        :type collection_name: str
        :param persist_directory: Directory to persist the collection.
        :type persist_directory: Optional[str]
        :param ids: List of document IDs, defaults to None.
        :type ids: Optional[List[str]]
        :param documents: List of documents to add to the vectorstore.
        :type documents: List[Document]
        :param embedding: Embedding function, defaults to None.
        :type embedding: Optional[Embeddings]
        :param client_settings: Chroma client settings
        :type client_settings: Optional[chromadb.config.Settings]
        :param client: Chroma client object, defaults to None.
        :type client: Optional[chromadb.Client]
        :return: Chroma vectorstore.
        :rtype: Chroma
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
        )