import pytest

import app.rag.vector_store as vector_store_module

from pathlib import Path
# ============================================================
# Fake Chroma Collection
# ============================================================


class FakeCollection:
    """
    Fake ChromaDB collection used for unit testing.
    """

    name = "documents"

    def __init__(self):
        self.add_calls = []
        self.query_calls = []
        self.get_calls = []
        self.delete_calls = []

        self.documents = []
        self.metadatas = []
        self.ids = []

    # --------------------------------------------------------
    # Chroma add
    # --------------------------------------------------------

    def add(
        self,
        ids,
        documents,
        embeddings,
        metadatas,
    ):
        self.add_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    # --------------------------------------------------------
    # Chroma count
    # --------------------------------------------------------

    def count(self):
        return len(self.ids)

    # --------------------------------------------------------
    # Chroma query
    # --------------------------------------------------------

    def query(self, **kwargs):
        self.query_calls.append(kwargs)

        return {
            "documents": [
                [
                    "First result",
                    "Second result",
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "resume.pdf",
                        "chunk": 1,
                    },
                    {
                        "source": "resume.pdf",
                        "chunk": 2,
                    },
                ]
            ],
            "distances": [
                [
                    0.1,
                    0.2,
                ]
            ],
        }

    # --------------------------------------------------------
    # Chroma get
    # --------------------------------------------------------

    def get(
        self,
        where=None,
        include=None,
        limit=None,
    ):
        self.get_calls.append(
            {
                "where": where,
                "include": include,
                "limit": limit,
            }
        )

        # --------------------------------------------------------
        # Determine available record count
        # --------------------------------------------------------

        record_count = max(
            len(self.ids),
            len(self.documents),
            len(self.metadatas),
        )

        indexes = list(
            range(record_count)
        )

        # --------------------------------------------------------
        # Filter by file hash
        # --------------------------------------------------------

        if where and "file_hash" in where:

            file_hash = where["file_hash"]

            matched_indexes = []

            for index in indexes:

                if index >= len(
                    self.metadatas
                ):
                    continue

                metadata = (
                    self.metadatas[index]
                )

                if (
                    metadata
                    and metadata.get("file_hash")
                    == file_hash
                ):
                    matched_indexes.append(
                        index
                    )

            indexes = matched_indexes

        # --------------------------------------------------------
        # Apply limit
        # --------------------------------------------------------

        if limit is not None:

            indexes = indexes[:limit]

        # --------------------------------------------------------
        # Build result safely
        # --------------------------------------------------------

        result = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        for index in indexes:

            if index < len(self.ids):

                result["ids"].append(
                    self.ids[index]
                )

            if index < len(self.documents):

                result["documents"].append(
                    self.documents[index]
                )

            if index < len(self.metadatas):

                result["metadatas"].append(
                    self.metadatas[index]
                )

        return result

    # --------------------------------------------------------
    # Chroma delete
    # --------------------------------------------------------

    def delete(self, ids):
        self.delete_calls.append(ids)

        remaining = []

        for index, document_id in enumerate(
            self.ids
        ):

            if document_id not in ids:

                remaining.append(
                    (
                        document_id,
                        self.documents[index],
                        self.metadatas[index],
                    )
                )

        self.ids = [
            item[0]
            for item in remaining
        ]

        self.documents = [
            item[1]
            for item in remaining
        ]

        self.metadatas = [
            item[2]
            for item in remaining
        ]


# ============================================================
# Fake Chroma Client
# ============================================================


class FakeChromaClient:

    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(
        self,
        name,
    ):
        return self.collection


# ============================================================
# Service Factory
# ============================================================


def create_vector_store(
    monkeypatch,
):
    """
    Create VectorStore with a fake ChromaDB backend.
    """

    collection = FakeCollection()

    client = FakeChromaClient(
        collection
    )

    monkeypatch.setattr(
        vector_store_module.chromadb,
        "PersistentClient",
        lambda path: client,
    )

    service = (
        vector_store_module.VectorStore()
    )

    return service, collection


# ============================================================
# Initialization
# ============================================================


def test_vector_store_initializes_successfully(
    monkeypatch,
):
    """
    VectorStore should initialize its ChromaDB
    client and collection.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    assert service.collection is collection
    assert service.collection.name == "documents"


def test_vector_store_rejects_missing_chroma_path(
    monkeypatch,
):
    """
    Missing CHROMA_DB_PATH should raise ValueError.
    """

    monkeypatch.setattr(
        vector_store_module.settings,
        "CHROMA_DB_PATH",
        "",
    )

    with pytest.raises(
        ValueError,
        match="CHROMA_DB_PATH is not configured.",
    ):

        vector_store_module.VectorStore()


# ============================================================
# Add Documents
# ============================================================


def test_add_documents_successfully(
    monkeypatch,
):
    """
    Valid chunks, embeddings and metadata should
    be stored in ChromaDB.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    chunks = [
        "First chunk",
        "Second chunk",
    ]

    embeddings = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    metadata = [
        {
            "source": "resume.pdf",
            "file_hash": "abc123",
            "chunk": 1,
        },
        {
            "source": "resume.pdf",
            "file_hash": "abc123",
            "chunk": 2,
        },
    ]

    service.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        metadata=metadata,
    )

    assert len(collection.add_calls) == 1

    call = collection.add_calls[0]

    assert call["documents"] == chunks
    assert call["embeddings"] == embeddings
    assert call["metadatas"] == metadata

    assert call["ids"] == [
        "abc123_1",
        "abc123_2",
    ]


def test_add_documents_rejects_empty_chunks(
    monkeypatch,
):
    """
    Empty chunks should raise ValueError.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="No chunks provided for indexing.",
    ):

        service.add_documents(
            chunks=[],
            embeddings=[],
            metadata=[],
        )


def test_add_documents_rejects_embedding_count_mismatch(
    monkeypatch,
):
    """
    Chunks and embeddings must have equal counts.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Chunks and embeddings count do not match.",
    ):

        service.add_documents(
            chunks=[
                "First",
                "Second",
            ],
            embeddings=[
                [0.1, 0.2],
            ],
            metadata=[
                {
                    "source": "test.txt",
                    "file_hash": "hash",
                    "chunk": 1,
                },
                {
                    "source": "test.txt",
                    "file_hash": "hash",
                    "chunk": 2,
                },
            ],
        )


def test_add_documents_rejects_metadata_count_mismatch(
    monkeypatch,
):
    """
    Chunks and metadata must have equal counts.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Chunks and metadata count do not match.",
    ):

        service.add_documents(
            chunks=[
                "First",
                "Second",
            ],
            embeddings=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            metadata=[
                {
                    "source": "test.txt",
                    "file_hash": "hash",
                    "chunk": 1,
                },
            ],
        )


def test_add_documents_rejects_missing_source(
    monkeypatch,
):
    """
    Metadata must contain a document source.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Metadata source missing",
    ):

        service.add_documents(
            chunks=["Test"],
            embeddings=[[0.1, 0.2]],
            metadata=[
                {
                    "file_hash": "hash",
                    "chunk": 1,
                }
            ],
        )


def test_add_documents_rejects_missing_file_hash(
    monkeypatch,
):
    """
    Metadata must contain a file hash.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Metadata file_hash missing",
    ):

        service.add_documents(
            chunks=["Test"],
            embeddings=[[0.1, 0.2]],
            metadata=[
                {
                    "source": "test.txt",
                    "chunk": 1,
                }
            ],
        )


def test_add_documents_rejects_invalid_chunk_number(
    monkeypatch,
):
    """
    Invalid chunk numbers should raise ValueError.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Invalid chunk number",
    ):

        service.add_documents(
            chunks=["Test"],
            embeddings=[[0.1, 0.2]],
            metadata=[
                {
                    "source": "test.txt",
                    "file_hash": "hash",
                    "chunk": "invalid",
                }
            ],
        )


def test_add_documents_rejects_non_positive_chunk_number(
    monkeypatch,
):
    """
    Chunk numbers must be greater than zero.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Chunk number must be greater than zero.",
    ):

        service.add_documents(
            chunks=["Test"],
            embeddings=[[0.1, 0.2]],
            metadata=[
                {
                    "source": "test.txt",
                    "file_hash": "hash",
                    "chunk": 0,
                }
            ],
        )


# ============================================================
# Search
# ============================================================


def test_search_successfully(
    monkeypatch,
):
    """
    Valid query embeddings should return ChromaDB results.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = [
        "id1",
        "id2",
    ]

    result = service.search(
        query_embedding=[0.1, 0.2],
        top_k=2,
    )

    assert "documents" in result
    assert "metadatas" in result
    assert "distances" in result

    assert len(
        collection.query_calls
    ) == 1

    query = collection.query_calls[0]

    assert query["query_embeddings"] == [
        [0.1, 0.2]
    ]

    assert query["n_results"] == 2


def test_search_rejects_empty_query_embedding(
    monkeypatch,
):
    """
    Empty query embeddings should raise ValueError.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Query embedding cannot be empty.",
    ):

        service.search(
            query_embedding=[],
            top_k=3,
        )


def test_search_rejects_invalid_top_k(
    monkeypatch,
):
    """
    top_k must be greater than zero.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero.",
    ):

        service.search(
            query_embedding=[0.1, 0.2],
            top_k=0,
        )


def test_search_returns_empty_result_when_collection_empty(
    monkeypatch,
):
    """
    Empty ChromaDB collections should return the
    standard empty search structure.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    result = service.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
    )

    assert result == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


def test_search_filters_by_single_document(
    monkeypatch,
):
    """
    A single document filter should create a
    source equality filter.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = ["id1"]

    service.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
        documents=["resume.pdf"],
    )

    query = collection.query_calls[0]

    assert query["where"] == {
        "source": "resume.pdf"
    }


def test_search_filters_by_multiple_documents(
    monkeypatch,
):
    """
    Multiple document filters should create
    a ChromaDB $or filter.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = ["id1"]

    service.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
        documents=[
            "resume.pdf",
            "project.pdf",
        ],
    )

    query = collection.query_calls[0]

    assert query["where"] == {
        "$or": [
            {
                "source": "resume.pdf"
            },
            {
                "source": "project.pdf"
            },
        ]
    }


def test_search_removes_duplicate_document_filters(
    monkeypatch,
):
    """
    Duplicate document names should be removed.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = ["id1"]

    service.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
        documents=[
            "resume.pdf",
            "resume.pdf",
            "project.pdf",
        ],
    )

    query = collection.query_calls[0]

    assert query["where"] == {
        "$or": [
            {
                "source": "resume.pdf"
            },
            {
                "source": "project.pdf"
            },
        ]
    }


def test_search_returns_empty_for_invalid_document_filter(
    monkeypatch,
):
    """
    A document filter containing no valid names should
    return the standard empty result.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = ["id1"]

    result = service.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
        documents=[
            "",
            "   ",
            None,
        ],
    )

    assert result == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    assert collection.query_calls == []


# ============================================================
# List Documents
# ============================================================


def test_list_documents_groups_chunks_by_file_hash(
    monkeypatch,
):
    """
    Multiple chunks belonging to the same file hash
    should produce one document summary.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.metadatas = [
        {
            "source": "resume.pdf",
            "file_type": "pdf",
            "file_hash": "hash1",
            "chunk": 1,
        },
        {
            "source": "resume.pdf",
            "file_type": "pdf",
            "file_hash": "hash1",
            "chunk": 2,
        },
        {
            "source": "project.pdf",
            "file_type": "pdf",
            "file_hash": "hash2",
            "chunk": 1,
        },
    ]

    result = service.list_documents()

    assert len(result) == 2

    resume = next(
        item
        for item in result
        if item["file_hash"] == "hash1"
    )

    assert resume["file_name"] == "resume.pdf"
    assert resume["file_type"] == "pdf"
    assert resume["chunks"] == 2


def test_list_documents_skips_invalid_metadata(
    monkeypatch,
):
    """
    Metadata without source or file hash should be ignored.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.metadatas = [
        {
            "source": "valid.txt",
            "file_hash": "hash1",
            "file_type": "txt",
        },
        {
            "source": "",
            "file_hash": "hash2",
        },
        {
            "source": "missing-hash.txt",
        },
        {},
        None,
    ]

    result = service.list_documents()

    assert len(result) == 1
    assert result[0]["file_name"] == "valid.txt"


def test_list_documents_sorts_by_file_name(
    monkeypatch,
):
    """
    Documents should be sorted by filename.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.metadatas = [
        {
            "source": "zebra.txt",
            "file_hash": "hash-z",
        },
        {
            "source": "alpha.txt",
            "file_hash": "hash-a",
        },
    ]

    result = service.list_documents()

    assert [
        item["file_name"]
        for item in result
    ] == [
        "alpha.txt",
        "zebra.txt",
    ]


# ============================================================
# Get Document
# ============================================================


def test_get_document_successfully(
    monkeypatch,
):
    """
    A valid file hash should return complete
    document details.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = [
        "hash1_2",
        "hash1_1",
    ]

    collection.documents = [
        "Second chunk",
        "First chunk",
    ]

    collection.metadatas = [
        {
            "source": "resume.pdf",
            "file_type": "pdf",
            "file_hash": "hash1",
            "chunk": 2,
        },
        {
            "source": "resume.pdf",
            "file_type": "pdf",
            "file_hash": "hash1",
            "chunk": 1,
        },
    ]

    result = service.get_document(
        "hash1"
    )

    assert result is not None

    assert result["file_name"] == "resume.pdf"
    assert result["file_type"] == "pdf"
    assert result["file_hash"] == "hash1"

    assert result["total_chunks"] == 2
    assert result["total_characters"] == (
        len("Second chunk")
        + len("First chunk")
    )

    assert [
        chunk["chunk"]
        for chunk in result["chunks"]
    ] == [
        1,
        2,
    ]


def test_get_document_returns_none_for_invalid_hash(
    monkeypatch,
):
    """
    Invalid hash values should return None.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    assert service.get_document(None) is None
    assert service.get_document("") is None
    assert service.get_document("   ") is None


def test_get_document_returns_none_when_not_found(
    monkeypatch,
):
    """
    Unknown hashes should return None.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = []

    result = service.get_document(
        "unknown"
    )

    assert result is None


# ============================================================
# Duplicate Detection
# ============================================================


def test_document_exists_returns_true(
    monkeypatch,
):
    """
    Existing file hashes should return True.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = ["hash1_1"]

    collection.metadatas = [
        {
            "source": "resume.pdf",
            "file_hash": "hash1",
        }
    ]

    result = service.document_exists(
        "hash1"
    )

    assert result is True


def test_document_exists_returns_false(
    monkeypatch,
):
    """
    Unknown file hashes should return False.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = []

    result = service.document_exists(
        "unknown"
    )

    assert result is False


def test_document_exists_returns_false_for_invalid_hash(
    monkeypatch,
):
    """
    Invalid hash values should return False.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    assert service.document_exists(None) is False
    assert service.document_exists("") is False
    assert service.document_exists("   ") is False


# ============================================================
# Statistics
# ============================================================


def test_get_statistics(
    monkeypatch,
):
    """
    Statistics should contain document and chunk counts.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.metadatas = [
        {
            "source": "a.txt",
            "file_hash": "hash-a",
        },
        {
            "source": "a.txt",
            "file_hash": "hash-a",
        },
        {
            "source": "b.txt",
            "file_hash": "hash-b",
        },
    ]

    result = service.get_statistics()

    assert result == {
        "documents": 2,
        "chunks": 3,
    }


def test_get_statistics_for_empty_store(
    monkeypatch,
):
    """
    Empty stores should report zero statistics.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    assert service.get_statistics() == {
        "documents": 0,
        "chunks": 0,
    }


# ============================================================
# Delete Document
# ============================================================


def test_delete_document_successfully(
    monkeypatch,
    tmp_path,
):
    """
    Existing documents should be removed from
    ChromaDB and their physical file should be deleted.
    """

    monkeypatch.setattr(
        vector_store_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    upload_file = (
        tmp_path / "resume.pdf"
    )

    upload_file.write_text(
        "test",
        encoding="utf-8",
    )

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = [
        "hash1_1",
        "hash1_2",
    ]

    collection.metadatas = [
        {
            "source": "resume.pdf",
            "file_hash": "hash1",
        },
        {
            "source": "resume.pdf",
            "file_hash": "hash1",
        },
    ]

    result = service.delete_document(
        "hash1"
    )

    assert result == {
        "file_name": "resume.pdf",
        "file_hash": "hash1",
        "deleted_chunks": 2,
        "deleted_file": True,
    }

    assert not upload_file.exists()

    assert collection.delete_calls == [
        [
            "hash1_1",
            "hash1_2",
        ]
    ]


def test_delete_document_returns_none_when_not_found(
    monkeypatch,
):
    """
    Unknown documents should return None.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = []

    result = service.delete_document(
        "unknown"
    )

    assert result is None


def test_delete_document_rejects_non_string_hash(
    monkeypatch,
):
    """
    File hash must be a string.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="File hash must be a string.",
    ):

        service.delete_document(
            None
        )


def test_delete_document_rejects_empty_hash(
    monkeypatch,
):
    """
    Empty hashes should raise ValueError.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="File hash cannot be empty.",
    ):

        service.delete_document(
            "   "
        )


# ============================================================
# Delete All Documents
# ============================================================


def test_delete_all_documents(
    monkeypatch,
    tmp_path,
):
    """
    All vectors and uploaded files should be deleted.
    """

    monkeypatch.setattr(
        vector_store_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    (tmp_path / "one.txt").write_text(
        "one",
        encoding="utf-8",
    )

    (tmp_path / "two.txt").write_text(
        "two",
        encoding="utf-8",
    )

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = [
        "hash1_1",
        "hash1_2",
        "hash2_1",
    ]

    collection.documents = [
        "one",
        "two",
        "three",
    ]

    collection.metadatas = [
        {
            "source": "one.txt",
            "file_hash": "hash1",
        },
        {
            "source": "one.txt",
            "file_hash": "hash1",
        },
        {
            "source": "two.txt",
            "file_hash": "hash2",
        },
    ]

    result = service.delete_all_documents()

    assert result == {
        "deleted_documents": 2,
        "deleted_chunks": 3,
    }

    assert collection.delete_calls == [
        [
            "hash1_1",
            "hash1_2",
            "hash2_1",
        ]
    ]

    assert not (
        tmp_path / "one.txt"
    ).exists()

    assert not (
        tmp_path / "two.txt"
    ).exists()


# ============================================================
# Database Info
# ============================================================


def test_get_database_info(
    monkeypatch,
):
    """
    Database info should contain collection name
    and statistics.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    result = service.get_database_info()

    assert result["collection"] == "documents"

    assert result["statistics"] == {
        "documents": 0,
        "chunks": 0,
    }


def test_show_collection(
    monkeypatch,
):
    """
    Collection information should contain name and count.
    """

    service, collection = (
        create_vector_store(monkeypatch)
    )

    collection.ids = [
        "id1",
        "id2",
        "id3",
    ]

    result = service.show_collection()

    assert result == {
        "collection": "documents",
        "count": 3,
    }


# ============================================================
# Empty Search Result
# ============================================================


def test_empty_search_result_structure():
    """
    The empty search helper should return a consistent
    ChromaDB-compatible structure.
    """

    result = (
        vector_store_module.VectorStore
        ._empty_search_result()
    )

    assert result == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    
    
    
# ============================================================
# Additional Coverage - Error Handling
# ============================================================


def test_vector_store_handles_chromadb_initialization_failure(
    monkeypatch,
):
    """ChromaDB initialization failure should propagate."""

    def failing_client(path):
        raise RuntimeError("ChromaDB unavailable")

    monkeypatch.setattr(
        vector_store_module.chromadb,
        "PersistentClient",
        failing_client,
    )

    with pytest.raises(
        RuntimeError,
        match="ChromaDB unavailable",
    ):
        vector_store_module.VectorStore()


def test_add_documents_handles_chromadb_failure(
    monkeypatch,
):
    """ChromaDB add failure should propagate."""

    service, collection = create_vector_store(
        monkeypatch
    )

    def failing_add(**kwargs):
        raise RuntimeError("add failed")

    monkeypatch.setattr(
        collection,
        "add",
        failing_add,
    )

    with pytest.raises(
        RuntimeError,
        match="add failed",
    ):
        service.add_documents(
            chunks=["Test"],
            embeddings=[[0.1, 0.2]],
            metadata=[
                {
                    "source": "test.txt",
                    "file_hash": "hash",
                    "chunk": 1,
                }
            ],
        )


def test_search_handles_chromadb_failure(
    monkeypatch,
):
    """ChromaDB query failure should propagate."""

    service, collection = create_vector_store(
        monkeypatch
    )

    collection.ids = ["id1"]

    def failing_query(**kwargs):
        raise RuntimeError("query failed")

    monkeypatch.setattr(
        collection,
        "query",
        failing_query,
    )

    with pytest.raises(
        RuntimeError,
        match="query failed",
    ):
        service.search(
            query_embedding=[0.1, 0.2],
            top_k=3,
        )


def test_search_returns_empty_when_chromadb_returns_no_documents(
    monkeypatch,
):
    """Search should return empty structure when Chroma returns no documents."""

    service, collection = create_vector_store(
        monkeypatch
    )

    collection.ids = ["id1"]

    def empty_query(**kwargs):
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    monkeypatch.setattr(
        collection,
        "query",
        empty_query,
    )

    result = service.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
    )

    assert result == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


def test_list_documents_handles_chromadb_failure(
    monkeypatch,
):
    """list_documents should propagate ChromaDB failures."""

    service, collection = create_vector_store(
        monkeypatch
    )

    def failing_get(**kwargs):
        raise RuntimeError("list failed")

    monkeypatch.setattr(
        collection,
        "get",
        failing_get,
    )

    with pytest.raises(
        RuntimeError,
        match="list failed",
    ):
        service.list_documents()


def test_get_document_handles_chromadb_failure(
    monkeypatch,
):
    """get_document should propagate ChromaDB failures."""

    service, collection = create_vector_store(
        monkeypatch
    )

    def failing_get(**kwargs):
        raise RuntimeError("get failed")

    monkeypatch.setattr(
        collection,
        "get",
        failing_get,
    )

    with pytest.raises(
        RuntimeError,
        match="get failed",
    ):
        service.get_document("hash1")


def test_delete_document_handles_chromadb_failure(
    monkeypatch,
):
    """delete_document should propagate ChromaDB failures."""

    service, collection = create_vector_store(
        monkeypatch
    )

    def failing_get(**kwargs):
        raise RuntimeError("delete lookup failed")

    monkeypatch.setattr(
        collection,
        "get",
        failing_get,
    )

    with pytest.raises(
        RuntimeError,
        match="delete lookup failed",
    ):
        service.delete_document("hash1")


def test_delete_document_handles_file_delete_failure(
    monkeypatch,
    tmp_path,
):
    """Physical file deletion failure should be handled."""

    monkeypatch.setattr(
        vector_store_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    upload_file = (
        tmp_path / "resume.pdf"
    )

    upload_file.write_text(
        "test",
        encoding="utf-8",
    )

    service, collection = create_vector_store(
        monkeypatch
    )

    collection.ids = ["hash1_1"]

    collection.metadatas = [
        {
            "source": "resume.pdf",
            "file_hash": "hash1",
        }
    ]

    original_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self == upload_file:
            raise OSError("permission denied")

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        failing_unlink,
    )

    result = service.delete_document(
        "hash1"
    )

    assert result == {
        "file_name": "resume.pdf",
        "file_hash": "hash1",
        "deleted_chunks": 1,
        "deleted_file": False,
    }


def test_document_exists_handles_chromadb_failure(
    monkeypatch,
):
    """document_exists should propagate ChromaDB failures."""

    service, collection = create_vector_store(
        monkeypatch
    )

    def failing_get(**kwargs):
        raise RuntimeError("exists check failed")

    monkeypatch.setattr(
        collection,
        "get",
        failing_get,
    )

    with pytest.raises(
        RuntimeError,
        match="exists check failed",
    ):
        service.document_exists("hash1")


def test_delete_all_documents_handles_chromadb_failure(
    monkeypatch,
):
    """delete_all_documents should propagate ChromaDB failures."""

    service, collection = create_vector_store(
        monkeypatch
    )

    def failing_get(**kwargs):
        raise RuntimeError("delete all failed")

    monkeypatch.setattr(
        collection,
        "get",
        failing_get,
    )

    with pytest.raises(
        RuntimeError,
        match="delete all failed",
    ):
        service.delete_all_documents()


def test_delete_all_documents_handles_non_file_entry(
    monkeypatch,
    tmp_path,
):
    """Directories inside upload folder should be ignored."""

    monkeypatch.setattr(
        vector_store_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    (tmp_path / "nested").mkdir()

    service, collection = create_vector_store(
        monkeypatch
    )

    collection.ids = []

    result = service.delete_all_documents()

    assert result == {
        "deleted_documents": 0,
        "deleted_chunks": 0,
    }

    assert (
        tmp_path / "nested"
    ).is_dir()


def test_delete_all_documents_handles_file_delete_failure(
    monkeypatch,
    tmp_path,
):
    """OSError while deleting an uploaded file should be handled."""

    monkeypatch.setattr(
        vector_store_module.settings,
        "UPLOAD_FOLDER",
        str(tmp_path),
    )

    upload_file = (
        tmp_path / "resume.pdf"
    )

    upload_file.write_text(
        "test",
        encoding="utf-8",
    )

    service, collection = create_vector_store(
        monkeypatch
    )

    collection.ids = ["hash1_1"]

    collection.documents = [
        "test",
    ]

    collection.metadatas = [
        {
            "source": "resume.pdf",
            "file_hash": "hash1",
        }
    ]

    original_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self == upload_file:
            raise OSError("permission denied")

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        failing_unlink,
    )

    result = service.delete_all_documents()

    assert result == {
        "deleted_documents": 1,
        "deleted_chunks": 1,
    }
    
def test_add_documents_rejects_missing_metadata(
    monkeypatch,
):
    """
    Missing metadata entries should raise ValueError.
    """

    service, _ = create_vector_store(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="Metadata missing for chunk 0",
    ):
        service.add_documents(
            chunks=["Test"],
            embeddings=[[0.1, 0.2]],
            metadata=[None],
        )
        
def test_search_returns_empty_when_documents_key_is_missing(
    monkeypatch,
):
    """
    Missing documents from ChromaDB should return
    the standard empty result.
    """

    service, collection = create_vector_store(
        monkeypatch
    )

    collection.ids = ["id1"]

    def empty_query(**kwargs):
        return {}

    monkeypatch.setattr(
        collection,
        "query",
        empty_query,
    )

    result = service.search(
        query_embedding=[0.1, 0.2],
        top_k=3,
    )

    assert result == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    
def test_delete_all_documents_handles_delete_failure(
    monkeypatch,
):
    """
    ChromaDB delete failure should be logged and
    propagated.
    """

    service, collection = create_vector_store(
        monkeypatch
    )

    collection.ids = [
        "hash1_1",
    ]

    collection.documents = [
        "test",
    ]

    collection.metadatas = [
        {
            "source": "test.txt",
            "file_hash": "hash1",
        }
    ]

    def failing_delete(ids):
        raise RuntimeError(
            "delete failed"
        )

    monkeypatch.setattr(
        collection,
        "delete",
        failing_delete,
    )

    with pytest.raises(
        RuntimeError,
        match="delete failed",
    ):
        service.delete_all_documents()