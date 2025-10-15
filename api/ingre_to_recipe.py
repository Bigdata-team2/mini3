


import pandas as pd
import json
from itertools import combinations



##df는 정제된 main 데이터 셋
### input_ingre 리스트현황


def filter_recipes(input_ingre, df):
    """
    주어진 input_ingre 리스트로 레시피를 필터링하여 JSON으로 반환.
    1) 모든 재료가 포함된 행 우선
    2) 없으면 한 재료 빠진 조합들 중 포함된 행 반환
    3) 그래도 없으면 빈 JSON 리스트 반환
    """

    # ① 모든 재료 포함된 행
    exact_df = df[df['rec1'].apply(lambda x: all(ing in x for ing in input_ingre))]
    if not exact_df.empty:
        return exact_df.to_json(orient='records', force_ascii=False, indent=2)

    # ② 모든 재료 포함된 행이 없을 경우 → 한 재료 빠진 조합들
    n = len(input_ingre)
    partial_dfs = []
    for combo in combinations(input_ingre, n - 1):
        combo_list = list(combo)
        matched = df[df['rec1'].apply(lambda x: all(ing in x for ing in combo_list))]
        if not matched.empty:
            partial_dfs.append(matched)

    # 조합 중 하나라도 매칭된 것이 있다면 DataFrame 합치기
    if partial_dfs:
        merged_df = pd.concat(partial_dfs).drop_duplicates().reset_index(drop=True)
        return merged_df.to_json(orient='records', force_ascii=False, indent=2)

    # ③ 아무 것도 없으면 빈 JSON 반환
    return json.dumps([], ensure_ascii=False)


if __name__ == "__main__":
    # 테스트용 예시 데이터


    # 예시 입력
    input_ingre = ['도토리묵', '상추', '당근', '양파','독']

    print(filter_recipes(input_ingre, df))