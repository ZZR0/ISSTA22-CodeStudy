import sys
import re

def filter_log(file_path):
    # 정규표현식: 숫자 뒤에 %| 가 오는 패턴 (예: 3%|, 100%|)
    pct_pattern = re.compile(r'(\d+)%\|')
    
    skipped_flag = False

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip()
                
                # 진행률 바(progress bar)인지 확인
                match = pct_pattern.search(line)
                
                if match and ("it/s]" in line or "s/it]" in line):
                    # % 숫자 추출
                    pct = int(match.group(1))
                    
                    # 0~10% 또는 90~100% 인 경우 -> 출력
                    if pct <= 10 or pct >= 90:
                        print(line)
                        skipped_flag = False # 다시 출력 모드로 전환
                    else:
                        # 11~89% 인 경우 -> 생략 (한 번만 표시)
                        if not skipped_flag:
                            print("   ... [ Middle Progress Skipped (11% ~ 89%) ] ...")
                            skipped_flag = True
                else:
                    # 진행률 바가 아닌 일반 로그(에러, 정보 등)는 무조건 출력
                    print(line)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_log.py <logfile>")
    else:
        filter_log(sys.argv[1])