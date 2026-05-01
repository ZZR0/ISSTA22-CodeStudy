# README-update.md (CodeBERT 실험 업데이트 정리) — 2026-02-05

이 문서는 `Clone-Detection-BigCloneBench/codebert/`에서 CodeBERT 실험을 재현/확장하기 위해,
오늘까지 적용한 변경사항과 실행 방법을 정리한다.

---

## 0) 디렉토리 구조

### codebert/
- `run.py` : 학습/평가 메인 스크립트
- `run-train.sh` : 학습 실행 스크립트(논문 재현 세팅 중심)
- `sample_data.py` : deterministic 10% 샘플(train/valid) 생성 스크립트
- `eval.sh`, `run.sh`, `model.py` : 기존 파일 유지

### dataset/ (상대경로: `../dataset/`)

../dataset/
├── data.jsonl
├── train.txt
├── valid.txt
├── test.txt
├── train_10percent.txt
├── valid_10percent.txt
├── bcb/
└── camel/

---

## 1) 문제/배경: run.py에 하드코딩된 10% 샘플링

원본 `run.py`에는 아래 코드가 있어 train/valid가 항상 10%만 사용될 수 있었다.

```python
if 'test' not in postfix:
    data = random.sample(data, int(len(data)*0.1))

이번 업데이트에서는 10% 샘플링을 코드 내부에서 하지 않고, 데이터 파일로 관리하는 방식으로 변경했다.
(실험 재현성과 데이터 관리 단순화를 위해)

⸻

2) 데이터 준비: deterministic + balanced 10% 샘플 생성

2.1 목적
	•	random seed 기반으로 매번 동일한 10% 샘플 생성 (deterministic)
	•	(옵션) label 0/1을 50:50으로 맞춘 balanced 샘플 생성

2.2 실행

codebert/ 디렉토리에서 실행:

python sample_data.py --seed 3 --mode balanced

생성 결과(예시):
	•	../dataset/train_10percent.txt
	•	../dataset/valid_10percent.txt

샘플링 크기:
	•	train: 901028 → 90102 (10%)
	•	valid: 415416 → 41540 (10%)

라벨 밸런스(예시):
	•	train_10percent: label0=45051 / label1=45051
	•	valid_10percent: label0=20770 / label1=20770

참고: --mode stratified는 원본 라벨 비율 유지, --mode balanced는 50:50 맞춤.

⸻

3) run.py 수정사항

3.1 내부 10% 샘플링 비활성화

TextDataset에서 아래 부분을 주석 처리하여, 외부에서 만든 10% 파일을 그대로 사용하도록 변경:

# NOTE: 10% sampling is handled externally (train_10percent.txt / valid_10percent.txt)
# if 'test' not in postfix:
#     data=random.sample(data,int(len(data)*0.1))

3.2 불필요한 파일 디스크립터 제거

미사용 open() 제거 (리소스 누수 방지):

# f=open(index_filename)  # unused file descriptor

3.3 데이터 로딩 수 로그 추가

실험 시 데이터가 의도대로 로딩됐는지 바로 확인할 수 있도록 로그 추가:

logger.info(f"[Dataset] postfix={postfix} pairs_loaded={len(data)}")


⸻

4) 학습 실행: run-train.sh

4.1 목적
	•	논문 재현 설정(paper setting): tokenizer는 roberta-base 사용
	•	train/valid는 외부에서 만든 10% 파일 사용

4.2 run-train.sh (요지)

# TOKENIZER=microsoft/codebert-base   # solution default (추후 최대 성능용)
TOKENIZER=roberta-base  # paper setting; original paper reproduction

TRAIN=../dataset/train_10percent.txt
VALID=../dataset/valid_10percent.txt
TEST=../dataset/test.txt

mkdir -p ./saved_models/

python run.py \
  --output_dir=./saved_models/ \
  --model_type=roberta \
  --config_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name=$TOKENIZER \
  --do_train \
  --do_test \
  --train_data_file=$TRAIN \
  --eval_data_file=$VALID \
  --test_data_file=$TEST \
  --epoch 2 \
  --block_size 400 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 3 2>&1 | tee ./saved_models/train.log

4.3 실행

bash run-train.sh


⸻

5) 로그 확인 포인트

5.1 데이터가 제대로 들어갔는지

학습 로그에서 아래 라인이 떠야 정상:
	•	train: pairs_loaded=90102
	•	valid: pairs_loaded=41540

grep -n "$begin:math:display$Dataset$end:math:display$" ./saved_models/train.log | head

5.2 학습 기본 요약

grep -nE "Running training|Num examples|Total optimization steps|Total train batch size|Gradient Accumulation" ./saved_models/train.log | head -n 30

5.3 Eval 결과

grep -n "Eval results" ./saved_models/train.log | tail -n 40


⸻

6) 향후 계획: 솔루션(최대 성능) 세팅

논문 재현이 끝나면, 솔루션 성능 최적화 목적의 추가 실험에서 tokenizer를 다음으로 변경 예정:
	•	TOKENIZER=microsoft/codebert-base

(Tokenizer만 바꾼 ablation run을 한 번 더 수행하여 성능 차이를 기록)

⸻

