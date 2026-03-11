# RAGAS 기반 RAG 파이프라인 평가 해설 (06 vs 05)

이 문서는 기존 규칙 기반(Rule-based) 평가 스크립트(`05_rag_tutorial_experiment_analysis.py`)와 비교하여 **모델 기반(LLM-as-a-judge) 평가 도구인 RAGAS**를 적용한 `06_ragas_evaluation_analysis.py`의 구현 차이점과 평가 방식의 이점을 상세히 해설합니다.

---

## 1. 평가 패러다임의 변화: 규칙 평가 vs 모델 평가

기존 05 파이프라인과 06 RAGAS 파이프라인의 가장 큰 차이점은 "채점 방식"입니다.

| 비교 항목 | 05 파이프라인 (규칙 기반) | 06 RAGAS 파이프라인 (모델 기반) |
| :--- | :--- | :--- |
| **정답 기준 (Reference)** | `"planning, memory, tool use"` 같은 **핵심 키워드 모음** | `"The main components are planning, memory, and tool use."` 같은 **완성된 자연어 문장** |
| **생성(Answer) 평가** | `answer_hit_rate`<br>(생성된 답변 안에 정답 키워드가 1개 이상 들어있는가?) | `Answer Relevancy` (주제 이탈 여부 파악)<br>`Faithfulness` (환각 발생 여부 파악) |
| **검색(Retrieval) 평가** | `hit_rate_at_k`<br>(검색된 문단 안에 정답 키워드가 들어있는가?) | `Context Precision` (정답 문단이 상위에 리스트업 되었는가?)<br>`Context Recall` (정답에 필요한 정보가 모두 검색되었는가?) |
| **평가자 (Judge)** | 단순 Python 코드의 `in` 연산자 (문자열 부분 일치 검사) | 강력한 LLM (여기서는 `gpt-4o-mini`)이 문맥을 직접 판단하여 스코어 부여 |

### 기존 규칙 평가(05)의 한계점
- **Lost in the middle 감지 불가**: 키워드가 하나라도 들어있으면 점수를 주기 때문에, LLM이 엉뚱한 맥락 속에서 단어만 언급해도 정답으로 만점 처리됩니다.
- **다양한 표현의 정답 처리 불가**: 정답이 "task decomposition" 인데 LLM이 "breaking down a task" 라고 답변하면, 의미는 완전히 같지만 프로그램은 이를 "오답" 처리합니다. (False Negative)

이러한 규칙 평가의 한계를 극복하기 위해 등장한 개념이 **"LLM을 채점관으로 쓰자(LLM-as-a-judge)"**는 것이며, 이를 규격화한 오픈소스 프레임워크가 바로 **RAGAS**입니다.

---

## 2. 코드 레벨의 구현 차이 (How to implement)

### 2-1. 데이터셋 준비 (EvaluationDataset)
RAGAS를 구동하기 위해서는 검색 결과와 생성 결과를 모아 `SingleTurnSample`이라는 객체 리스트로 만들어야 합니다.

```python
# 06_ragas_evaluation_analysis.py의 핵심 데이터 준비 로직
from ragas import EvaluationDataset, SingleTurnSample

samples = []
for case in EVALUATION_SET:
    # 1. 문서 검색 및 답변 생성
    answer, retrieved_contexts = get_rag_outputs(vector_store, config, case.query)
    
    # 2. RAGAS 전용 Sample 객체로 포장
    sample = SingleTurnSample(
        user_input=case.query,
        retrieved_contexts=retrieved_contexts, # 검색기가 찾은 문단들 리스트
        response=answer,                       # LLM이 최종적으로 한 대답
        reference=case.reference,              # 우리가 기대하는 자연어 모범 답안
    )
    samples.append(sample)

# 3. 데이터셋으로 변환
dataset = EvaluationDataset(samples=samples)
```

### 2-2. 평가 모델(LLM Judge) 설정 및 실행
RAGAS는 내부적으로 OpenAI API 통신을 수행하기 때문에 LLM 객체와 Embedding 객체를 초기화해서 넘겨주어야 합니다.

```python
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

# 1. 평가용 모델 세팅 (일반적으로 GPT-4 계열 권장)
ragas_llm = llm_factory("gpt-4o-mini", client=client)
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# 2. 사용할 4대 Metrics 지정
metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]

# 3. 평가 실행 (이 단계에서 LLM API 호출이 다량 발생함)
result = evaluate(dataset=dataset, metrics=metrics)
```

---

## 3. RAGAS의 4대 핵심 평가지표 해설 (Metrics)

실험결과인 `06_ragas_experiment_summary.csv`를 보면 다음 네 줄의 점수대(0.0 ~ 1.0)가 나타납니다.

### 1) Faithfulness (신뢰성 / 환각 방지)
- **질문**: "LLM의 답변이 **오직 검색해 온 문단(`retrieved_contexts`) 내에서만** 작성되었는가?"
- **왜 중요한가**: RAG의 존재 이유입니다. LLM이 아는 척하며 외부 지식을 지어내면(Hallucination) 이 점수가 급락합니다. 점수가 낮다면 시스템 프롬프트("컨텍스트 외의 내용은 말하지 마라")를 더 강력하게 수정해야 합니다.

### 2) Answer Relevancy (답변의 관련성)
- **질문**: "LLM의 답변이 사용자의 **원래 질문(`user_input`)에 정확히 초점을** 맞추고 있는가?"
- **어떻게 채점하나**: 흥미롭게도 RAGAS는 LLM이 만든 대답을 거꾸로 읽고 "어떤 질문을 받았을까?"를 역추산(Reverse-engineering)하여, 원래 사용자의 질문과 의미적 벡터 유사도를 비교합니다. 동문서답을 하면 점수가 떨어집니다.

### 3) Context Precision (검색 정밀도)
- **질문**: "정답과 관련된 핵심 정보가 **상위 순위(Top-K 안에서도 맨 위)**에 잘 노출되었는가?"
- **왜 중요한가**: 검색된 문단 중 1위에 쓸데없는 문단이 껴있고 3위에 정답이 있다면 점수가 깎입니다. 이 점수가 낮다면 Chunking 전략을 수정하거나 Reranker(재정렬 모델)를 도입해야 합니다.

### 4) Context Recall (검색 재현율)
- **질문**: "**모범 답안(`reference`)에 있는 모든 정보**를 검색 단계(`retrieved_contexts`)에서 전부 찾아왔는가?"
- **왜 중요한가**: 이 점수는 평가자가 만든 모범 답안이 있어야만 측정됩니다. 정답을 말하기 위해 3가지 정보가 필요한데 검색기가 2개만 찾아왔다면 점수가 깎입니다. 이 점수가 낮다면 `retrieval_k` 개수를 늘리거나 Vector DB의 검색 알고리즘을 손봐야 합니다.

---

## 4. 실무 관점의 RAGAS 도입 조언

RAGAS는 매우 강력하지만 "은탄환"은 아닙니다. 현업 도입 시 다음 사항을 주의해야 합니다.

1. **비용(Cost)과 속도(Latency)**: 기존 파이프라인(05)은 채점에 거의 0초, 0원이 들었지만, RAGAS(06)는 모든 질의의 평가마다 여러 번의 LLM 프롬프트가 뒷단에서 오가기 때문에 비용과 시간이 크게 늘어납니다. (대규모 데이터셋 전체에는 `05` 방식을, 그 중 골라낸 샘플링 셋에만 RAGAS `06` 방식을 적용하는 하이브리드 전략을 추천합니다.)
2. **모범 답안(Reference)의 강제성**: `Context Precision` 스코어를 제대로 받으려면 사람이 일일이 고품질의 자연어 `reference`를 작성해 주어야 합니다. 데이터셋을 구축하는 휴먼 리소스가 핵심 관건입니다.

---

> [!TIP]
> 이제 `python3 06_ragas_evaluation_analysis.py` 스크립트를 직접 실행하시어 분석된 CSV 데이터와 플롯을 확보하신 뒤, `05` 실험에서의 결과와 **RAGAS가 잡아낸 미묘한 점수 차이**를 눈으로 직접 확인해 보시기 바랍니다!
