import requests
from PIL import Image
import base64
from io import BytesIO

# 定义服务器地址
SERVER_URL = "http://localhost:5000"

# 图像转Base64工具函数
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 添加物品到服务端
def add_item(item_data):
    response = requests.post(
        f"{SERVER_URL}/items",
        json=item_data
    )
    if response.status_code != 200:
        raise Exception(f"添加物品失败: {response.json()['error']}")
    return response.json()

# 计算兼容性分数
def compute_score():
    response = requests.post(f"{SERVER_URL}/compute")
    if response.status_code != 200:
        raise Exception(f"计算分数失败: {response.json()['error']}")
    return response.json()["score"]

if __name__ == "__main__":
    # 准备测试数据
    items = [
        {
            "image": pil_to_base64(Image.open("datasets/polyvore/images/117427809.jpg")),
            "description": "Tie front rayon shirt",
            "category": "tops"
        },
        {
            "image": pil_to_base64(Image.open("datasets/polyvore/images/197823931.jpg")),
            "description": "peacock feather appliques backpack",
            "category": "bags"
        }
    ]

    # 清空原有物品（可选）
    # requests.delete(f"{SERVER_URL}/items/clear")  # 假设有清空端点

    # 添加所有物品
    for item in items:
        print(f"添加物品: {item['description']}")
        add_item(item)

    # 计算兼容性分数
    try:
        score = compute_score()
        print(f"兼容性分数：{score:.4f}")
    except Exception as e:
        print(f"错误: {str(e)}")