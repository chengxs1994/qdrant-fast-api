# 介绍

提供了基于 qdrant 向量数据库的学习用例和通过 fastapi 搭建高性能搜索服务API Demo.

### 快速导航
- 常用向量数据操作用例：tests/test_qdrant.py
- 搜索服务API：service.py
- 异步支持：neural_searcher.py

## 环境要求

- pip 23.1.0+
- Python 3.8.10+

## 安装

### qdrant
```shell
docker run -p 6333:6333 qdrant/qdrant
```

### 项目
```shell
# dev 环境
pip install -r requirements.txt
```

## 启动

tests/test_qdrant.py 为 qdrant 的学习用例。可直接单例执行。

官方入门文档：https://qdrant.tech/documentation/quick-start/

```shell
python service.py
```

启动后访问 http://127.0.0.1:8000/docs 即可查看接口文档。