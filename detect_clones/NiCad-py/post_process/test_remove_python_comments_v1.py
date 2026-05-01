import io
import tokenize
from remove_python_comments import remove_python_comments

def test_filtering_scenarios():
    scenarios = {
        "Azure_Style_Docstring": """
def __init__(
    self,
    **kwargs
):
    \"\"\"
    This docstring should be removed.
    \"\"\"
    super().__init__(**kwargs)
    self.val = kwargs.get('val')
""",
        "Simple_Boilerplate": """
def empty_func(self):
    pass
""",
        "Real_Logic": """
def calculate(a, b):
    \"\"\"Docstring to remove\"\"\"
    if a > b:
        return a + b
    return b - a
""",
        # 추가된 실제 사례: Azure SDK의 자동 생성된 __init__ 보일러플레이트
        "Azure_SDK_Boilerplate": """
    def __init__(
        self,
        **kwargs
    ):
        \"\"\"
        :keyword capacity: If the SKU supports scale out/in then the capacity integer should be
         included. If scale out/in is not possible for the resource this may be omitted.
        :paramtype capacity: int
        :keyword family: If the service has different generations of hardware, for the same SKU, then
         that can be captured here.
        :paramtype family: str
        \"\"\"
        super(PartialSku, self).__init__(**kwargs)
        self.capacity = kwargs.get('capacity', None)
        self.family = kwargs.get('family', None)
        self.name = kwargs.get('name', None)
        self.size = kwargs.get('size', None)
        self.tier = kwargs.get('tier', None)
"""
    }

    print("=== Filter Test Results ===")
    for name, code in scenarios.items():
        # 1. 주석 및 Docstring 제거 수행
        cleaned = remove_python_comments(code)
        
        # 2. 결과 분석 (LOC 계산 등)
        non_empty_lines = [ln for ln in cleaned.splitlines() if ln.strip()]
        clean_loc = len(non_empty_lines)
        
        print(f"[{name}]")
        print(f"Cleaned Code (Pure Logic):\n{cleaned.strip()}")
        print(f"Resulting LOC: {clean_loc}")
        
        # 보일러플레이트 판별 로직 예시 (향후 Tree-sitter 도입 시 검증용)
        if clean_loc < 7 and "super" in cleaned:
            print(">> Status: Potential Boilerplate (High similarity with other Inits)")
        
        print("-" * 40)

if __name__ == "__main__":
    test_filtering_scenarios()