# crawler_to_chroma.py
import requests
from lxml import html
import pandas as pd
import json
from main import build_chroma

def crawl_and_save_recipe(recipe_id: int):
    """레시피 크롤링 후 ChromaDB에 저장"""
    url = f"https://www.10000recipe.com/recipe/{recipe_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    tree = html.fromstring(response.content)
    
    # 요리명 추출
    recipe_name = tree.xpath('//h3[@class="view2_summary_info3"]/text()')
    recipe_name = recipe_name[0].strip() if recipe_name else f"레시피_{recipe_id}"
    
    # 재료 추출
    ingredients = tree.xpath('//div[@class="ready_ingre3"]//li/text()')
    ingredients_text = " ".join([ing.strip() for ing in ingredients])
    
    # 조리 단계 추출
    recipe_steps = []
    for i in range(1, 20):
        step = tree.xpath(f'//*[@id="stepdescr{i}"]/text()')
        if step:
            for text in step:
                text_cleaned = text.strip()
                if text_cleaned:
                    recipe_steps.append({
                        "recipe_id": str(recipe_id),
                        "recipe_num": i,
                        "recipe_text": text_cleaned,
                        "recipe_img_url": ""
                    })
        else:
            break
    
    # 데이터 구성
    data = {
        "recipes_info": [{
            "recipe_id": str(recipe_id),
            "요리명": recipe_name,
            "요리별재료": ingredients_text,
            "조리시간": "30분",
            "난이도": "중급",
            "카테고리": "한식"
        }],
        "recipes_steps": recipe_steps,
        "user_allergy": []  # 사용자 알러지 정보는 별도로 추가
    }
    
    return data

# 사용 예시
if __name__ == "__main__":
    # 여러 레시피 크롤링
    recipe_ids = [1942052, 1942053, 1942054]  # 예시 ID들
    
    all_data = {
        "recipes_info": [],
        "recipes_steps": [],
        "user_allergy": []
    }
    
    for recipe_id in recipe_ids:
        try:
            data = crawl_and_save_recipe(recipe_id)
            all_data["recipes_info"].extend(data["recipes_info"])
            all_data["recipes_steps"].extend(data["recipes_steps"])
            print(f"레시피 {recipe_id} 크롤링 완료")
        except Exception as e:
            print(f"레시피 {recipe_id} 크롤링 실패: {e}")
    
    # ChromaDB에 저장
    # build_chroma(all_data)  # API 서버가 실행 중일 때