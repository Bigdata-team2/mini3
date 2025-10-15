# build_recipes.py
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

client = chromadb.Client(Settings(persist_directory="./chroma_data"))
embed = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
col = client.get_or_create_collection("recipes", embedding_function=embed)

# 예시 데이터 (실제로는 CSV/DB에서 읽어와 일괄 upsert)
docs = [
    "Tomato Egg Stir-fry: tomato, egg, salt, oil",
    "Simple Omelette: egg, milk, salt, pepper",
    "Pasta Aglio e Olio: spaghetti, garlic, olive oil, chili",
]
metas = [
    {"recipe_id": "R001", "title": "토마토 달걀 볶음", "ingredients": ["tomato","egg"], "time": 10},
    {"recipe_id": "R002", "title": "오믈렛", "ingredients": ["egg","milk"], "time": 7},
    {"recipe_id": "R003", "title": "알리오 올리오", "ingredients": ["garlic","olive oil","spaghetti"], "time": 15},
]
ids = ["R001","R002","R003"]

col.upsert(documents=docs, metadatas=metas, ids=ids)
print("✅ 레시피 upsert 완료")
