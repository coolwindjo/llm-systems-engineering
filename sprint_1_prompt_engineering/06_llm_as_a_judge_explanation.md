# LLM-as-a-Judge 실습 설명

이 문서는 `06_practice_llm_as_a_judge.py`가 어떤 방식으로 커리어 상담 챗봇 응답을 자동 평가하는지 설명합니다.  
핵심 아이디어는 "모델이 생성한 답변을 또 다른 LLM이 심사하게 만든다"는 것입니다.

## 1) 실습 목표

이 실습의 목표는 프롬프트 품질을 정량적으로 비교하는 것입니다.

- 비교 대상: `A_zero_shot`, `B_few_shot`, `C_custom`
- 테스트 데이터: 다양한 난이도/카테고리의 커리어 질문
- 평가 방법: Judge 프롬프트로 4개 축 점수화

이 접근을 사용하면 "어떤 프롬프트가 더 좋다"를 감(느낌) 대신 점수와 근거로 판단할 수 있습니다.

## 2) 구성 요소

### 프롬프트 후보 (`PROMPTS`)

서로 다른 스타일의 시스템 프롬프트를 사전에 정의합니다.

- `A_zero_shot`: 짧고 단순한 지시
- `B_few_shot`: 예시를 포함한 구조화 응답 유도
- `C_custom`: 강한 페르소나 + 오프토픽 리다이렉트 규칙

### 테스트셋 (`TEST_CASES`)

각 문항은 다음 메타정보를 포함합니다.

- `question`: 사용자 질문
- `category`: 문제 유형
- `difficulty`: 난이도
- `key_aspects`: 좋은 답변이 포함해야 할 핵심 요소

`key_aspects`는 Judge가 단순 문장 품질이 아니라 "질문 요구사항 충족도"를 평가하게 만드는 기준점입니다.

### Judge 템플릿 (`JUDGE_PROMPT_TEMPLATE`)

Judge는 아래 4개 축을 1~5점으로 평가합니다.

- `Coherence`: 논리적 구조
- `Relevance`: 질문 및 핵심 요소 반영
- `Fluency`: 문장 자연스러움/전문성
- `Consistency`: 내부 모순 여부

출력 형식을 고정해 파싱 가능성을 높였습니다.

## 3) 핵심 함수 흐름

### `get_career_advice(...)`

챗봇 역할 모델을 호출해 실제 답변을 생성합니다.

### `evaluate_response(...)`

생성된 답변과 `key_aspects`를 Judge 프롬프트에 넣어 평가 응답을 받습니다.

### `parse_judge_output(...)`

Judge의 자유 텍스트를 점수 딕셔너리로 변환합니다.

- `REASONING:` 구간 추출
- 각 축(`Coherence` 등) 점수/근거 분해
- 파싱 실패 시 `parse_error=True`, 기본 점수(3)로 폴백

이 폴백 덕분에 일부 형식 이탈이 있어도 전체 실험이 중단되지 않습니다.

## 4) 실행 모드

스크립트는 CLI 모드를 제공합니다.

- `smoke`: 각 프롬프트의 샘플 응답 확인
- `single`: 특정 문항 1개를 Judge로 채점
- `eval`: 전체(또는 일부) 데이터셋 일괄 평가
- `all`: 위 모드를 순차 실행

예시:

```bash
python sprint_1_prompt_engineering/06_practice_llm_as_a_judge.py --mode single --variant B_few_shot --question-id 6
python sprint_1_prompt_engineering/06_practice_llm_as_a_judge.py --mode eval --limit 4
python sprint_1_prompt_engineering/06_practice_llm_as_a_judge.py --mode eval --plot
python sprint_1_prompt_engineering/06_practice_llm_as_a_judge.py --mode eval --plot --plot-path sprint_1_prompt_engineering/results_llm_judge.png
```

## 5) 결과 해석 포인트

`eval` 모드에서는 프롬프트별 평균 점수와 종합 점수(`Overall`)를 출력합니다.

해석 시에는 아래를 함께 보세요.

- 평균 점수: 전반적 성능 우위 확인
- `parse_error` 개수: Judge 출력 안정성 확인
- 카테고리/난이도별 편차: 특정 상황 취약점 진단

## 결론

`06_practice_llm_as_a_judge.py`는 프롬프트 엔지니어링을 "생성 → 심사 → 집계" 루프로 자동화하는 기본 골격입니다.  
이 구조를 확장하면 모델 비교, 회귀 테스트, 릴리스 전 품질 게이트까지 연결할 수 있습니다.
