import json
import sys
import textwrap
# 기존 로직 import
from remove_python_comments import remove_python_comments

def test_cleaning_logic(jsonl_path):
    print(f"Reading: {jsonl_path}")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sources = data.get("sources", [])
            
            for src in sources:
                original_code = src.get("code", "")
                print(f"\nTarget File: {src.get('file')}")
                print(f"Original:\n {original_code}")
                print("-" * 60)

                # --- Case 1: 기존 방식 (Raw Input) ---
                print("[Test 1] 기존 로직 그대로 실행 (Raw Input)")
                cleaned_1 = remove_python_comments(original_code)
                print(f"Updated:\n {cleaned_1}")
                
                if '"""' in cleaned_1 or "Sends a POST request" in cleaned_1:
                    print("❌ 결과: Docstring 제거 실패!")
                    print("   이유: 함수 앞의 들여쓰기(Indent) 때문에 tokenize가 코드를 잘못 해석함.")
                else:
                    print("✅ 결과: 성공")

                print("-" * 60)

                # --- Case 2: 수정 제안 (With Dedent) ---
                print("[Test 2] 들여쓰기 보정 후 실행 (textwrap.dedent)")
                # 코드 전체의 공통된 들여쓰기를 제거하여 'def'를 라인 맨 앞으로 당김
                dedented_code = textwrap.dedent(original_code)
                cleaned_2 = remove_python_comments(dedented_code)
                
                if '"""' in cleaned_2 or "Sends a POST request" in cleaned_2:
                    print("❌ 결과: 여전히 실패")
                else:
                    print("✅ 결과: Docstring 제거 성공!")
                    print("   확인: 코드가 왼쪽으로 정렬되어 tokenize가 정상 작동함.")
                    
                print("-" * 60)
                print("[Final Output Preview - Test 2 Result]")
                # 결과 코드의 앞부분만 출력하여 확인
                print(cleaned_2.strip()[:300] + "\n...")

if __name__ == "__main__":
    # 파일이 없으면 생성 여부 확인 필요
    test_cleaning_logic("./data/small-input.jsonl")