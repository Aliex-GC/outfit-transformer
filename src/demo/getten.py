import os
import base64
import pickle
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import torch
import pathlib
from argparse import Namespace

# 假设已有模块
from .vectorstore import FAISSVectorStore
from ..models.load import load_model
from ..data.datasets import polyvore

app = Flask(__name__)

# 常量定义
POLYVORE_CATEGORIES = [
    'all-body', 'bottoms', 'tops', 'outerwear', 'bags',
    'shoes', 'accessories', 'scarves', 'hats',
    'sunglasses', 'jewellery', 'unknown'
]

class RecommendationSystem:
    def __init__(self, args):
        # 加载元数据
        self.metadata = polyvore.load_metadata(args.polyvore_dir)
        
        # 加载预计算嵌入
        self.embedding_dict = self._load_precomputed(args.polyvore_dir)
        
        # 建立元数据索引
        self.id_to_meta = self._build_metadata_index(args.polyvore_dir)
        
        # 初始化FAISS索引
        self.indexer = FAISSVectorStore(
            index_name='rec_index',
            d_embed=128,
            faiss_type='IndexFlatIP',
            base_dir=os.path.join(args.polyvore_dir, "precomputed_rec_embeddings"),
        )

        
        # 加载特征提取模型
        self.model = load_model(
            model_type='clip',  # 根据实际模型类型调整
            checkpoint='checkpoints/compatibillity_clip_best.pth'
        ).eval()
        
        # 建立类别索引
        self.category_index = self._build_category_index()

    def _load_precomputed(self, polyvore_dir):
        """加载预计算嵌入"""
        e_dir = os.path.join(polyvore_dir, "precomputed_rec_embeddings")
        embeddings = {}
        for filename in os.listdir(e_dir):
            if filename.endswith(".pkl"):
                with open(os.path.join(e_dir, filename), 'rb') as f:
                    data = pickle.load(f)
                    for item_id, emb in zip(data['ids'], data['embeddings']):
                        embeddings[item_id] = emb
        return embeddings

    def _build_metadata_index(self, polyvore_dir):
        """构建元数据索引"""
        dataset = polyvore.PolyvoreItemDataset(
            polyvore_dir,
            metadata=self.metadata,
            load_image=False  # 不需要实际加载图像
        )
        return {item.item_id: item for item in dataset}

    def _build_category_index(self):
        """按类别索引物品ID"""
        index = {c: [] for c in POLYVORE_CATEGORIES}
        for item_id, meta in self.id_to_meta.items():
            category = meta.semantic_category.lower()
            index.get(category, index['unknown']).append(item_id)
        return index

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """提取查询图像特征"""
        with torch.no_grad():
            processed = self.model.preprocess(image).unsqueeze(0)
            return self.model(processed).cpu().numpy().flatten()

@app.route('/search', methods=['POST'])
def search_similar():
    """
    请求格式：
    {
        "image": "base64字符串",
        "target_category": "outerwear"
    }
    """
    data = request.json
    
    # 验证输入
    if not data.get("image") or not data.get("target_category"):
        return jsonify({"error": "Missing required fields"}), 400
    
    if data["target_category"] not in POLYVORE_CATEGORIES:
        return jsonify({"error": f"Invalid category. Valid options: {POLYVORE_CATEGORIES}"}), 400

    try:
        # 转换图像
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        
        # 提取特征
        query_embedding = app.config['REC_SYS'].extract_features(image)
        
        # 获取目标类别物品ID
        target_ids = app.config['REC_SYS'].category_index[data["target_category"]]
        
        # 执行搜索
        scores, item_ids = app.config['REC_SYS'].indexer.search(
            query_embedding.reshape(1, -1),
            k=min(100, len(target_ids))  # 先取前100再过滤
        )
        # 过滤并排序
        results = []
        seen = set()
        for score, item_id in zip(scores[0], item_ids[0]):
            if item_id in target_ids and item_id not in seen:
                meta = app.config['REC_SYS'].id_to_meta[item_id]
                results.append({
                    "item_id": item_id,
                    "score": float(score),
                    "image_path": f"images/{item_id}.jpg",  # 根据实际路径调整
                    "description": meta.description,
                    "category": meta.semantic_category,
                    "metadata": {
                        "title": meta.title,
                        "related": meta.related
                    }
                })
                seen.add(item_id)
                if len(results) >= 10:
                    break
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return jsonify({"results": results[:10]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def initialize_system(polyvore_dir="./datasets/polyvore"):
    args = Namespace(
        polyvore_dir=polyvore_dir
    )
    app.config['REC_SYS'] = RecommendationSystem(args)

if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=5000)