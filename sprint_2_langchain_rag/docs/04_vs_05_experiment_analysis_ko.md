# RAG 실험 결과 비교 분석: 04 (Knowledge Base) vs 05 (RAG Tutorial, 개선 후)

이 문서는 다음 결과를 기준으로 작성했습니다.

- [04_knowledge_base_experiment_summary.csv](/workspace/sprint_2_langchain_rag/analysis_outputs/04_knowledge_base_experiment_summary.csv)
- [05_rag_tutorial_experiment_summary.csv](/workspace/sprint_2_langchain_rag/analysis_outputs/05_rag_tutorial_experiment_summary.csv)
- [05_rag_tutorial_query_details.csv](/workspace/sprint_2_langchain_rag/analysis_outputs/05_rag_tutorial_query_details.csv)

핵심 전제는 하나입니다.

- **이전의 05 실험은 너무 쉬웠다.**
- **지금의 05 실험은 harder query set, stricter relevance rule, answer-level evaluation을 포함하도록 개선되었다.**

따라서 이 문서는 예전처럼 "`05는 아무 설정을 해도 다 맞는다`"는 결론을 반복하지 않습니다. 대신, **개선 후 05가 실제로 어느 정도 변별력을 갖게 되었는지**를 04와 비교해 해설합니다.

---

## 1. 한눈에 보는 결론

### 04 실험

04는 여전히 가장 강한 retrieval stress test입니다.

- `top1_accuracy`: `0.5000 ~ 0.8333`
- `hit_rate_at_k`: `0.8333 ~ 1.0000`
- `mrr`: `0.6389 ~ 0.8889`

즉, chunking이나 overlap 같은 retrieval 변인을 건드리면 성능이 실제로 크게 흔들립니다.

### 05 실험

개선 후 05는 더 이상 포화된 테스트가 아닙니다.

- `top1_accuracy`: `0.5714 ~ 0.7143`
- `hit_rate_at_k`: `0.7143 ~ 0.8571`
- `mrr`: `0.6905 ~ 0.7143`
- `answer_hit_rate`: `0.5714 ~ 0.7143`
- `avg_answer_keyword_coverage`: `0.6190 ~ 0.7143`

즉, 05도 이제 설정 변화에 따라 retrieval와 answer가 함께 흔들립니다. 다만, **흔들리는 방식이 04와는 다릅니다.**

> 요약:
> - **04**는 "chunking이 retrieval 자체를 어떻게 무너뜨리는가"를 보기 좋은 테스트
> - **05**는 "retrieval 차이가 최종 answer까지 어떻게 번지는가"를 보기 좋아진 테스트

---

## 2. 05 실험에서 무엇이 달라졌는가

이번 개선에서 중요한 변경은 세 가지였습니다.

### 1. Query가 harder해졌다

예전 05는 `"What are the main components..."`처럼 사실상 섹션 제목과 거의 같은 질문이 많았습니다. 지금은 그런 질문 대신 아래처럼 **특정 프레임워크나 benchmark를 직접 겨냥하는 질문**으로 바뀌었습니다.

- Chain of Thought
- Tree of Thoughts
- Reflexion
- ReAct
- APIBank
- Gorilla

이제는 단순히 `Planning`, `Memory`, `Tool use` 같은 큰 제목만 맞춰서는 안 됩니다.

### 2. Relevance 판정이 더 엄격해졌다

예전에는 keyword 하나만 들어 있어도 relevant로 처리될 수 있었습니다. 지금은 query마다 **최소 2개 keyword match**를 요구합니다.

이 변화는 중요합니다.

- 예전: 관련 단어 하나만 걸려도 정답 취급
- 지금: 실제로 더 구체적인 문맥까지 맞아야 정답 취급

즉, retrieval의 질을 더 엄격하게 재게 되었습니다.

### 3. Generation 평가가 기본이 되었다

이제 `05_rag_tutorial_experiment_analysis.py`는 항상 answer-level 평가를 수행합니다.

- `answer_hit_rate`
- `avg_answer_keyword_coverage`

게다가 이전에는 `two_step` 체인에서 `retrieval_k` 변경이 answer 단계에 제대로 반영되지 않았는데, 지금은 이 부분도 고쳐졌습니다. 따라서 `k=5`가 실제 answer 품질에 어떤 영향을 주는지 볼 수 있게 되었습니다.

---

## 3. 새 05 결과는 어떻게 읽어야 하는가

### `chunk_size_1500`

retrieval 기준으로는 가장 안정적인 쪽입니다.

- `hit_rate_at_k = 0.8571`
- `mrr = 0.7143`

즉, **더 큰 청크가 특정 개념 설명을 한 덩어리로 보존하는 데 유리**했습니다. 블로그 포스트처럼 구조화된 문서에서도, 세부 프레임워크를 묻는 harder query에서는 큰 청크가 여전히 도움이 된다는 뜻입니다.

### `retrieval_k_5`

이번 05에서는 `k=5`가 실제로 의미 있는 개선을 보였습니다.

- `answer_hit_rate = 0.7143`로 최고
- `avg_answer_keyword_coverage = 0.7143`로 최고
- `avg_relevant_count = 1.5714`로 최고

즉, 05는 이제 **"정답 chunk를 더 많이 확보하면 answer가 실제로 좋아지는"** 실험이 됐습니다. 예전처럼 `k`를 바꿔도 차이가 없던 상태와는 분명히 달라졌습니다.

### `chunk_size_500`

이 설정은 흥미롭습니다.

- `top1_accuracy = 0.7143`로 가장 높음
- 하지만 `hit_rate_at_k = 0.7143`로 가장 낮음

즉, **맞을 때는 1등으로 잘 맞추지만, 한 번 틀리면 회복력이 약한** 패턴입니다.

사용자 관점에서 보면:

- 검색 UI에서는 첫 결과가 또렷해서 좋아 보일 수 있습니다.
- 하지만 RAG 시스템 전체로 보면, 실패한 query를 top-k 안에서 회복하는 능력은 약합니다.

### `baseline_agentic`

agentic이 retrieval 자체를 개선하지는 않았습니다.

- retrieval 스코어는 `baseline_two_step`과 동일
- 하지만 `avg_answer_keyword_coverage`는 `0.6667`로 `baseline_two_step`의 `0.6190`보다 약간 높음

즉, 이 데이터셋에서는 에이전트가 **검색을 더 잘하는 것은 아니지만**, answer phrasing이나 tool usage 과정에서 약간 더 풍부한 답을 만들 가능성은 보입니다. 다만 그 차이는 아직 크지 않습니다.

### `embedding_large`

이 설정은 여전히 큰 차이를 만들지 못했습니다.

- retrieval 스코어는 baseline과 동일
- answer coverage만 소폭 상승

즉, **이 단계에서는 임베딩 모델 업그레이드보다 query 설계와 retrieval depth가 더 큰 영향**을 줬습니다.

---

## 4. 질의별로 보면 무엇이 어려워졌는가

05의 query details를 보면, 이제 일부 질문은 진짜로 변별력을 만들고 있습니다.

### 잘 맞는 질문

다음 질문들은 대부분 설정에서 안정적으로 맞습니다.

- Chain of Thought
- long-term memory + vector store
- APIBank

이 질문들은 특정 키워드 조합이 한 chunk 안에 비교적 선명하게 모여 있어서 retrieval이 안정적입니다.

### 헷갈리는 질문

다음 질문들은 05를 어렵게 만드는 핵심 포인트입니다.

- Reflexion 관련 질문
- Gorilla 관련 질문

특히 Gorilla 질문은 여러 설정에서 `I do not know`가 나오거나, TALM/Toolformer로 잘못 답하는 경우가 보입니다. 이건 좋은 현상입니다. 왜냐하면 **이제 05도 hard negative를 가진 테스트가 되었기 때문**입니다.

Reflexion 질문도 자주 `Chain of Hindsight`로 오답이 납니다. 즉, self-improvement라는 상위 개념만 맞고, 정확한 하위 프레임워크 구분에는 실패하고 있습니다.

이런 종류의 실패는 실제 RAG에서 더 중요합니다. 사용자는 보통 "planning이 뭐야?"보다 "그중 어떤 방법이 이런 역할을 하느냐?"처럼 더 좁은 질문을 하기 때문입니다.

---

## 5. 04와 05는 이제 어떻게 역할이 다른가

### 04의 역할

04는 retrieval 튜닝용 벤치마크로 가장 좋습니다.

- 논문이 조밀하고
- 용어가 반복되고
- 청크를 조금만 잘못 잘라도 의미가 깨지고
- overlap, chunk_size, model 선택이 실제 점수 차이로 바로 드러납니다

즉, **retrieval 파이프라인의 기초 체력을 보는 데 최적**입니다.

### 05의 역할

개선 후 05는 answer-aware RAG 벤치마크로 더 유용해졌습니다.

- retrieval만이 아니라 answer까지 평가함
- `two_step` vs `agentic` 비교가 가능함
- `k=5`가 실제 answer 품질을 끌어올리는지 확인 가능함
- 특정 프레임워크 수준에서 오답과 혼동이 발생함

즉, **05는 이제 "retrieval 차이가 실제 답변 품질에 어떻게 번지는가"를 보여주는 실험**으로 보는 것이 맞습니다.

---

## 6. 실무 의사결정 관점 해석

### retrieval 튜닝이 목적이면

우선 04를 봐야 합니다.

- chunking
- overlap
- embedding model

이 세 변수의 민감도가 가장 선명하게 드러납니다.

### answer 품질과 pipeline 선택이 목적이면

05를 같이 봐야 합니다.

- `k=5`가 answer를 실제로 개선하는가
- `agentic`이 `two_step`보다 나은가
- 더 큰 embedding model이 answer를 개선하는가

이 질문들은 04보다 05가 더 잘 보여줍니다.

### 현재 결과 기준 추천

- retrieval 기본 튜닝: 04 중심
- RAG pipeline 비교: 05 중심
- 비용 대비 효과: `embedding_large`보다 query quality와 `k` 조정이 우선
- answer 개선: `retrieval_k_5`가 현재 05에서 가장 실질적인 개선을 보여줌

---

## 7. 최종 결론

이제 05는 더 이상 "너무 쉬워서 아무 의미 없는 실험"이 아닙니다.

개선 후의 05는 다음 역할을 갖습니다.

- **04**: retrieval 민감도와 chunking 효과를 보는 강한 stress test
- **05**: harder query + answer evaluation으로 pipeline 효과를 보는 실전형 보조 벤치마크

한 문장으로 정리하면 이렇습니다.

- **04는 retrieval를 부수기 좋은 벤치마크이고, 05는 retrieval 차이가 answer로 번지는지 보기 좋은 벤치마크가 되었다.**
