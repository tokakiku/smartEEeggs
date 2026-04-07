from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGEngine:

    def __init__(self):
        # 加载embedding模型
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # 示例知识库
        self.docs = [
            "TCP三次握手用于建立客户端和服务器之间的连接。",
            "HTTP是应用层协议，用于传输网页数据。",
            "IP协议负责网络层的数据传输。",
            "DNS用于将域名解析为IP地址。"
        ]
        # 向量化
        embeddings = self.model.encode(self.docs)

        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings))

    def search(self, query, top_k=2):
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector), top_k)
        results = []
        for idx in indices[0]:
            results.append(self.docs[idx])
        return results
if __name__ == "__main__":
    rag = RAGEngine()

    result = rag.search("什么是TCP连接")

    print(result)