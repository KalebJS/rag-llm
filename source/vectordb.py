import os

from fastapi import HTTPException
from pinecone import Pinecone, PodSpec
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

PINECONE_ENVIRONMENT: str = os.environ.get("PINECONE_ENV", "gcp-starter")
TOP_K_DOCUMENTS = 3
INDEX_NAME = "document-indexer"

openapi_config = OpenApiConfiguration.get_default_copy()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


if INDEX_NAME not in {x["name"] for x in pc.list_indexes()}:
    pc.create_index(name=INDEX_NAME, metric="cosine", dimension=1024, spec=PodSpec(environment=PINECONE_ENVIRONMENT))

index = pc.Index(INDEX_NAME)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def add_document_to_db(document_id: str, paragraphs: list[str], embeddings: list[str]):
    try:
        labeled_embeddings = [
            (
                f"{document_id}_{i}",  # Id of vector
                embedding,  # Dense vector values
                {"document_id": document_id, "sentence_id": i, "text": paragraph}
                # For ease of architecture I will save the text in pinecone as well
                # This is not recommended since Pinecone memory might be expensive
            )
            for i, (paragraph, embedding) in enumerate(zip(paragraphs, embeddings))
        ]
        for embedding_chunk in chunks(labeled_embeddings, 100):
            index.upsert(vectors=embedding_chunk)
    except Exception as e:
        raise HTTPException(404, detail=f"Pinecone indexing fetch fail with error {e}")


def fetch_top_paragraphs(document_id: str, embedding: list[float]) -> list[str]:
    try:
        query_response = index.query(
            top_k=TOP_K_DOCUMENTS,
            vector=embedding,
            filter={
                "document_id": {"$eq": document_id},
            },
            include_metadata=True,
        )
    except Exception as e:
        raise HTTPException(404, detail=f"Pinecone indexing fetch fail with error {e}")

    answers = [q["metadata"]["text"] for q in query_response["matches"]]
    return answers
