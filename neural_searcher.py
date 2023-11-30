from qdrant_client import QdrantClient, AsyncQdrantClient
from sentence_transformers import SentenceTransformer

import asyncio

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.qdrant_client = QdrantClient("http://localhost:6333")

    # 查询
    def search(self, text: str):
        vector = self.model.encode(text).tolist()

        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5
        )

        payloads = [hit.payload for hit in search_result]
        return payloads

    # 异步查询
    async def async_search(self, text: str):
        # AsyncQdrantClient提供与同步对应项相同的方法QdrantClient，异步客户端是在qdrant-client1.6.1版本中引入
        client = AsyncQdrantClient("http://localhost:6333")

        vector = self.model.encode(text).tolist()

        search_result = await client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5
        )

        payloads = [hit.payload for hit in search_result]
        return payloads