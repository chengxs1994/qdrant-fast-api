import unittest
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
import torch
import torchvision.transforms as transforms
from PIL import Image
class TestQDrantImg(unittest.TestCase):

    def setUp(self):
        self.collection_name = "img_collection"
        self.client = QdrantClient("localhost", port=6333)
        # 加载ResNet-50模型
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.model.eval()

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 添加集合
    def test_create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1000, distance=Distance.EUCLID),
        )

    # 添加图片向量
    def test_img_vector(self):
        # 加载并预处理图像
        id = 1
        image_path = './img/cat1.png'
        # id = 2
        # image_path = './img/dog1.png'
        image = Image.open(image_path)
        image_tensor = self.preprocess(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        with torch.no_grad():
            feature_vector = self.model(image_tensor).squeeze().tolist()

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=[{'id': id, 'vector': feature_vector, 'payload': {"image_path": image_path}}]
        )

        print(operation_info)

    # 匹配图片向量
    def test_search(self):
        # 加载并预处理图像
        image_path = './img/cat2.png'
        # image_path = './img/dog1.png'
        # image_path = './img/cat3.png'
        image = Image.open(image_path)
        image_tensor = self.preprocess(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        with torch.no_grad():
            feature_vector = self.model(image_tensor).squeeze().tolist()

        search_result = self.client.search(
            collection_name=self.collection_name, query_vector=feature_vector, limit=3
            , with_vectors=True, with_payload=True
        )

        print(search_result)