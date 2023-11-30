import unittest
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import models, QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams

from sentence_transformers import SentenceTransformer

import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm


class TestQDrant(unittest.TestCase):

    def setUp(self):
        self.client = QdrantClient("localhost", port=6333)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------------- 向量入门 ----------------------

    # 添加集合
    def test_create(self):
        self.client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=4, distance=Distance.DOT),
        )

    # 添加向量
    def test_upsert(self):
        operation_info = self.client.upsert(
            collection_name="test_collection",
            wait=True,
            points=[
                PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
                PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
                PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
                PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
                PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
                PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
            ],
        )

        print(operation_info)

    # 运行查询
    def test_search(self):
        search_result = self.client.search(
            collection_name="test_collection", query_vector=[0.2, 0.1, 0.9, 0.7], limit=3
            , with_vectors=True, with_payload=True
        )

        print(search_result)

    # 添加过滤器
    def test_query_filter(self):

        search_result = self.client.search(
            collection_name="test_collection",
            query_vector=[0.2, 0.1, 0.9, 0.7],
            query_filter=Filter(
                must=[FieldCondition(key="city", match=MatchValue(value="London"))]
            ),
            search_params=SearchParams(ef=100, ef_search=100),
            limit=3,
            score_threshold=0.9
        )

        print(search_result)

    # ---------------------- 数据集转向量 ----------------------

    # 数据集转向量
    def test_models_embeddings(self):
        # 指定模型
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # 添加数据集
        documents = [
            {
                "name": "The Time Machine",
                "description": "A man travels through time and witnesses the evolution of humanity.",
                "author": "H.G. Wells",
                "year": 1895,
            },
            {
                "name": "Ender's Game",
                "description": "A young boy is trained to become a military leader in a war against an alien race.",
                "author": "Orson Scott Card",
                "year": 1985,
            },
            {
                "name": "Brave New World",
                "description": "A dystopian society where people are genetically engineered and conditioned to conform to a strict social hierarchy.",
                "author": "Aldous Huxley",
                "year": 1932,
            },
            {
                "name": "The Hitchhiker's Guide to the Galaxy",
                "description": "A comedic science fiction series following the misadventures of an unwitting human and his alien friend.",
                "author": "Douglas Adams",
                "year": 1979,
            },
            {
                "name": "Dune",
                "description": "A desert planet is the site of political intrigue and power struggles.",
                "author": "Frank Herbert",
                "year": 1965,
            },
            {
                "name": "Foundation",
                "description": "A mathematician develops a science to predict the future of humanity and works to save civilization from collapse.",
                "author": "Isaac Asimov",
                "year": 1951,
            },
            {
                "name": "Snow Crash",
                "description": "A futuristic world where the internet has evolved into a virtual reality metaverse.",
                "author": "Neal Stephenson",
                "year": 1992,
            },
            {
                "name": "Neuromancer",
                "description": "A hacker is hired to pull off a near-impossible hack and gets pulled into a web of intrigue.",
                "author": "William Gibson",
                "year": 1984,
            },
            {
                "name": "The War of the Worlds",
                "description": "A Martian invasion of Earth throws humanity into chaos.",
                "author": "H.G. Wells",
                "year": 1898,
            },
            {
                "name": "The Hunger Games",
                "description": "A dystopian society where teenagers are forced to fight to the death in a televised spectacle.",
                "author": "Suzanne Collins",
                "year": 2008,
            },
            {
                "name": "The Andromeda Strain",
                "description": "A deadly virus from outer space threatens to wipe out humanity.",
                "author": "Michael Crichton",
                "year": 1969,
            },
            {
                "name": "The Left Hand of Darkness",
                "description": "A human ambassador is sent to a planet where the inhabitants are genderless and can change gender at will.",
                "author": "Ursula K. Le Guin",
                "year": 1969,
            },
            {
                "name": "The Three-Body Problem",
                "description": "Humans encounter an alien civilization that lives in a dying system.",
                "author": "Liu Cixin",
                "year": 2008,
            },
        ]

        # 创建集合
        self.client.recreate_collection(
            collection_name="my_books",
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        )

        # 上传数据到集合
        self.client.upload_records(
            collection_name="my_books",
            records=[
                models.Record(
                    id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
                )
                for idx, doc in enumerate(documents)
            ],
        )

    # 向引擎询问问题
    def test_models_embeddings_search(self):
        hits = self.client.search(
            collection_name="my_books",
            query_vector=self.model.encode("alien invasion").tolist(),
            limit=3,
        )
        for hit in hits:
            print(hit.payload, "score:", hit.score)

    # 添加过滤器
    def test_models_embeddings_filter(self):
        hits = self.client.search(
            collection_name="my_books",
            query_vector=self.model.encode("alien invasion").tolist(),
            query_filter=models.Filter(
                must=[models.FieldCondition(key="year", range=models.Range(gte=2000))]
            ),
            limit=1,
        )
        for hit in hits:
            print(hit.payload, "score:", hit.score)

    # ---------------------- json数据集转向量 ----------------------

    # json数据集转向量
    def test_upload_json_npy(self):
        # 样本数据集
        # wget https://storage.googleapis.com/generall-shared-data/startups_demo.json
        df = pd.read_json("./startups_demo.json", lines=True)

        # 对所有启动描述进行编码，为每个启动描述创建一个嵌入向量。在内部，该encode函数会将输入分成批次，这将显著加快该过程。
        vectors = self.model.encode(
            [row.alt + ". " + row.description for row in df.itertuples()],
            show_progress_bar=True,
        )
        # 将保存的向量下载到名为的新文件中startup_vectors.npy
        np.save("startup_vectors.npy", vectors, allow_pickle=False)

        self.client.recreate_collection(
            collection_name="startups",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # 迭代取json数据源，作为向量有效负载
        fd = open("./startups_demo.json")
        payload = map(json.loads, fd)
        fd.close()

        # 将所有向量加载到内存中
        vectors = np.load("./startup_vectors.npy")

        self.client.upload_collection(
            collection_name="startups",
            vectors=vectors,
            payload=payload,
            ids=None,  # 自动分配ID
            batch_size=256,  # 每批次矢量数大小
        )

        print(vectors.shape)  # > (40474, 384) 有 40474 个 384 维的向量

    # 从向量库搜索
    def test_upload_json_npy_search(self):
        text = "Artificial intelligence machine learning"
        vector = self.model.encode(text).tolist()

        search_result = self.client.search(
            collection_name="startups",
            query_vector=vector,
            query_filter=None,
            limit=5
        )

        payloads = [hit.payload for hit in search_result]

        print(payloads)

    # 添加过滤器
    def test_upload_json_npy_filter(self):
        text = "Artificial intelligence machine learning"
        vector = self.model.encode(text).tolist()

        city_of_interest = "Berlin"

        # Define a filter for cities
        city_filter = Filter(**{
            "must": [{
                "key": "city",  # Store city information in a field of the same name
                "match": {  # This condition checks if payload field has the requested value
                    "value": city_of_interest
                }
            }]
        })

        search_result = self.client.search(
            collection_name="startups",
            query_vector=vector,
            query_filter=city_filter,
            limit=5
        )

        payloads = [hit.payload for hit in search_result]

        print(payloads)
