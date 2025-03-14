import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from dataclasses import dataclass
from typing import List

# 项目模块导入
from .vectorstore import FAISSVectorStore
from ..models.load import load_model
from ..data import datatypes
from ..data.datasets import polyvore
POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"
# 全局状态容器
class AppState:
    def __init__(self):
        self.my_items = []
        self.polyvore_dataset = None
        self.compatibility_model = None
        self.complementary_model = None
        self.indexer = None

state = AppState()
app = Flask(__name__)
CORS(app)

# 工具函数
def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_pil(base64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_str)))

# API 端点
@app.route('/items', methods=['POST'])
def handle_items():
    if request.method == 'POST':
        data = request.json
        try:
            if not all(k in data for k in ["image", "description", "category"]):
                return jsonify({"error": "Missing required fields"}), 400

            new_item = datatypes.FashionItem(
                id=None,
                image=base64_to_pil(data["image"]),
                description=data["description"],
                category=data["category"]
            )
            state.my_items.append(new_item)
            return jsonify({"message": "Item added", "count": len(state.my_items)}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/items/<int:index>', methods=['DELETE'])
def remove_item(index):
    try:
        if 0 <= index < len(state.my_items):
            del state.my_items[index]
            return jsonify({"message": "Item deleted"}), 200
        return jsonify({"error": "Invalid index"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/polyvore', methods=['GET'])
def get_polyvore_page():
    try:
        page = int(request.args.get('page', 1))
        per_page = 12
        start = (page-1)*per_page
        end = start + per_page
        
        return jsonify({
            "items": [
                {
                    "image": pil_to_base64(item.image),
                    "description": item.description,
                    "category": item.category
                }
                for item in state.polyvore_dataset[start:end]
            ]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compute', methods=['POST'])
def compute_compatibility():
    try:
        if not state.my_items:
            return jsonify({"error": "No items to compute"}), 400

        query = datatypes.FashionCompatibilityQuery(outfit=state.my_items)
        with torch.no_grad():
            score = state.compatibility_model.predict_score(
                query=[query], 
                use_precomputed_embedding=False
            )[0].item()
        
        return jsonify({"score": round(score, 4)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search_complementary():
    try:
        query = datatypes.FashionComplementaryQuery(
            outfit=state.my_items,
            category='Unknown'
        )
        
        with torch.no_grad():
            embedding = state.complementary_model.embed_query(
                query=[query],
                use_precomputed_embedding=False
            ).cpu().numpy()

        results = state.indexer.search(embedding, k=8)[0]
        
        return jsonify({
            "results": [
                pil_to_base64(state.polyvore_dataset.get_item_by_id(item_id).image)
                for _, item_id in results
            ]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 初始化函数
def initialize_backend(
    model_type: str = "clip",
    compatibility_checkpoint: str = "checkpoints/compatibillity_clip_best.pth",
    complementary_checkpoint: str = "checkpoints/complementary_clip_best.pth",
    polyvore_dir: str = "./datasets/polyvore"
):
    """初始化后端服务
    
    Args:
        model_type: 模型类型 (original/clip)
        compatibility_checkpoint: 兼容性模型检查点路径
        complementary_checkpoint: 互补模型检查点路径 
        polyvore_dir: Polyvore数据集目录
    """
    # 加载数据集
    metadata = polyvore.load_metadata(polyvore_dir)
    state.polyvore_dataset = polyvore.PolyvoreItemDataset(
        polyvore_dir,
        metadata=metadata,
        load_image=True
    )
    
    # 初始化模型
    state.compatibility_model = load_model(
        model_type=model_type,
        checkpoint=compatibility_checkpoint
    ).eval()
    
    state.complementary_model = load_model(
        model_type=model_type,
        checkpoint=complementary_checkpoint
    ).eval()
    
    # 初始化FAISS
    state.indexer = FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(
            polyvore_dir=polyvore_dir
        ),
    )

def start_server(host: str = "0.0.0.0", port: int = 5000):
    """启动服务"""
    app.run(host=host, port=port)

if __name__ == '__main__':
    # 使用示例
    initialize_backend(
        model_type="clip",
        compatibility_checkpoint="checkpoints/compatibillity_clip_best.pth",
        complementary_checkpoint="checkpoints/complementary_clip_best.pth",
        polyvore_dir="./datasets/polyvore"
    )
    start_server()