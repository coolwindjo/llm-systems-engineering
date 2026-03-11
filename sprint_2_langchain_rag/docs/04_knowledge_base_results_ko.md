# Knowledge Base 실험 결과 해설

이 문서는 다음 결과 파일을 바탕으로 작성했습니다.

- [04_knowledge_base_experiment_summary.csv](/workspace/sprint_2_langchain_rag/analysis_outputs/04_knowledge_base_experiment_summary.csv)
- [04_knowledge_base_query_details.csv](/workspace/sprint_2_langchain_rag/analysis_outputs/04_knowledge_base_query_details.csv)
- [04_knowledge_base_overall_scores.png](/workspace/sprint_2_langchain_rag/analysis_outputs/04_knowledge_base_overall_scores.png)
- [04_knowledge_base_query_heatmap.png](/workspace/sprint_2_langchain_rag/analysis_outputs/04_knowledge_base_query_heatmap.png)
- [04_knowledge_base_chunk_tradeoffs.png](/workspace/sprint_2_langchain_rag/analysis_outputs/04_knowledge_base_chunk_tradeoffs.png)

분석 대상 PDF는 `attention_is_all_you_need.pdf`이며, 비교한 변수는 다음 다섯 가지였습니다.

- `chunk_size`: `500`, `1000`, `1500`
- `chunk_overlap`: `200`, `50`
- retrieval `k`: `3`, `5`
- embedding model: `text-embedding-3-small`, `text-embedding-3-large`
- 기본 설정 대비 각 변수 하나씩만 변경

## 0. 지표를 어떻게 읽어야 하는가

이 문서의 숫자는 단순한 모델 점수표가 아니라, "사용자가 실제로 검색 결과를 봤을 때 어떤 경험을 하게 되는가"를 간접적으로 보여주는 신호입니다.

### `top1_accuracy`

`top1_accuracy`는 첫 번째 검색 결과가 바로 쓸 만한 답을 담고 있을 확률에 가깝게 해석할 수 있습니다.

- 값이 높을수록 사용자는 첫 결과만 읽어도 원하는 정보를 얻을 가능성이 높습니다.
- 챗봇, 검색창, 사내 문서 QA처럼 사용자가 긴 리스트를 잘 읽지 않는 제품에서 특히 중요합니다.

사용자 관점에서는 이렇게 이해하면 됩니다.

- `top1_accuracy`가 높다: "첫 답변이 대체로 맞는다"
- `top1_accuracy`가 낮다: "검색은 되는 것 같은데, 첫 결과가 자주 빗나간다"

### `hit_rate_at_k`

`hit_rate_at_k`는 상위 `k`개 안에 정답에 가까운 문서가 하나라도 들어왔는지를 봅니다.

- 값이 높으면, 시스템이 완전히 엉뚱한 방향으로 가지는 않는다는 뜻입니다.
- 다만 첫 결과가 틀리고 2~5번째에만 정답이 있더라도 높게 나올 수 있습니다.

사용자 관점에서는 이렇게 해석하는 것이 실용적입니다.

- `hit_rate_at_k`가 높다: "조금만 더 찾아보면 정답이 있다"
- `hit_rate_at_k`가 낮다: "상위 몇 개를 봐도 답이 없을 가능성이 높다"

이 지표는 사람이 직접 결과 리스트를 훑는 검색 서비스에는 중요하지만, LLM이 상위 몇 개만 받아 자동으로 답변하는 RAG에서는 단독으로 믿기 어렵습니다. 왜냐하면 정답이 3번째에만 있어도, LLM이 1번째와 2번째 문맥에 더 크게 끌릴 수 있기 때문입니다.

### `mrr`

`mrr`는 "첫 번째 관련 문서가 얼마나 앞쪽 순위에 나타나는가"를 측정합니다.

- 1위면 `1.0`
- 2위면 `0.5`
- 3위면 `0.333...`
- 아예 없으면 `0.0`

즉, `mrr`는 `hit_rate_at_k`보다 더 엄격하게 순위 품질을 봅니다.

사용자 관점에서는 다음처럼 읽는 것이 좋습니다.

- `mrr`가 높다: "정답이 있으면 대체로 앞에 나온다"
- `mrr`가 낮다: "정답이 있더라도 뒤쪽에 숨어 있다"

RAG에서는 이 지표가 매우 중요합니다. 이유는 LLM이 앞쪽 문맥의 영향을 더 많이 받기 쉽기 때문입니다. `hit_rate_at_k`가 같아도 `mrr`가 높은 설정이 실제 답변 품질에서 더 나을 가능성이 큽니다.

### `relevant_count`

`relevant_count`는 상위 `k`개 결과 안에 관련 청크가 몇 개나 들어왔는지를 의미합니다.

- 값이 높으면 비슷한 근거 문서가 여러 개 들어왔다는 뜻입니다.
- 하지만 이것이 항상 좋은 것은 아닙니다.

사용자 관점에서 중요한 포인트는 다음입니다.

- `relevant_count`가 높다: "근거 후보가 많이 모인다"
- 하지만 너무 높으면: "비슷한 내용이 중복으로 많이 들어와 문맥 낭비가 생길 수 있다"

특히 RAG에서는 관련 청크가 많아도 모두 비슷한 문장을 반복하면 실제 답변 품질에는 큰 도움이 안 될 수 있습니다. 그래서 이 지표는 보조 지표로 보는 것이 맞습니다.

### 어떤 지표를 가장 우선해야 하는가

사용 시나리오별 우선순위는 보통 아래처럼 잡는 것이 현실적입니다.

1. 사용자가 첫 답변을 바로 신뢰해야 하는 챗봇/QA: `top1_accuracy`, `mrr`
2. 사용자가 결과 목록을 직접 읽는 검색 서비스: `hit_rate_at_k`, `mrr`
3. 근거 문서를 여러 개 모아 요약하는 시스템: `hit_rate_at_k`, `relevant_count`, 그 다음 `mrr`

이번 실험처럼 RAG를 염두에 둔 상황에서는 `top1_accuracy`와 `mrr`를 더 중요하게 보는 것이 합리적입니다.

## 1. 한눈에 보는 결론

가장 좋은 전체 성능은 `chunk_size_1500`이었습니다.

- `top1_accuracy`: `0.8333`
- `hit_rate_at_k`: `1.0000`
- `mrr`: `0.8889`

즉, 큰 청크로 나누었을 때 관련 정보가 한 청크 안에 함께 묶일 가능성이 높아졌고, 그 결과 질의별 첫 검색 결과의 품질도 좋아졌습니다.

반대로 가장 약한 설정은 `chunk_size_500`이었습니다.

- `top1_accuracy`: `0.5000`
- `hit_rate_at_k`: `0.8333`
- `mrr`: `0.6389`

청크를 너무 잘게 나누면, 문맥이 여러 조각으로 찢어지면서 검색은 되더라도 상위 1개 결과의 정확성이 크게 떨어진다는 점을 보여줍니다.

## 2. 설정별 해석

### `chunk_size_1500`

가장 균형이 좋았습니다.

- 청크 수가 `38개`로 줄어들어 검색 후보가 단순해졌습니다.
- `long-range dependencies` 질의까지 top-1에서 맞히면서 전체 `hit_rate_at_k`가 `1.0`이 되었습니다.
- 다만 청크가 커질수록 답변 생성 단계에서 불필요한 내용도 함께 들어갈 수 있으므로, RAG 최종 응답 품질까지 항상 최고라고 단정할 수는 없습니다.

### `baseline` (`chunk_size=1000`, `chunk_overlap=200`, `k=3`, `small`)

전반적으로 안정적이지만, 가장 어려운 질의를 끝내 해결하지는 못했습니다.

- `top1_accuracy`: `0.8333`
- `hit_rate_at_k`: `0.8333`
- `mrr`: `0.8333`

특히 `Why is the Transformer good at modeling long-range dependencies?` 질의에서 관련 키워드를 포함한 청크를 top-3 안에 찾지 못했습니다.

### `retrieval_k_5`

직관적으로는 더 많은 문서를 보면 성능이 좋아질 것 같지만, 이번 지표에서는 거의 개선이 없었습니다.

- `top1_accuracy`: baseline과 동일
- `hit_rate_at_k`: baseline과 동일
- `mrr`: baseline과 동일
- `avg_relevant_count`: `2.17 -> 3.33`으로 증가

즉, 관련 문서의 총량은 더 많이 가져오지만, 첫 관련 문서의 순위 자체는 좋아지지 않았습니다. 검색 단계보다 재정렬(reranking)이나 프롬프트 설계가 필요한 상황으로 볼 수 있습니다.

사용자 관점의 해석은 더 중요합니다.

- 만약 사용자가 직접 검색 결과 5개를 읽는다면, 이 설정은 나쁘지 않습니다.
- 하지만 LLM이 상위 문맥 몇 개만 받아 자동으로 답하는 구조라면, `relevant_count` 증가만으로는 체감 개선이 거의 없을 수 있습니다.

즉, "`k`를 늘리면 더 좋아질 것"이라는 직관은 절반만 맞습니다. 정답을 더 많이 포함할 수는 있지만, 더 앞에 올려주지는 못했습니다.

### `embedding_large`

더 큰 임베딩 모델이 항상 더 좋은 결과를 주지는 않았습니다.

- `top1_accuracy`: baseline과 동일
- `hit_rate_at_k`: baseline과 동일
- `mrr`: baseline과 동일

다만 질의별 세부 결과를 보면 일부 점수는 더 높아졌습니다. 예를 들어 positional encoding, multi-head attention 관련 쿼리에서 top result score는 약간 좋아졌습니다. 하지만 전체 순위 메트릭은 바뀌지 않았습니다.

정리하면, 이 실험에서는 모델 크기 증가보다 청크 전략이 더 큰 영향을 주었습니다.

실제 의사결정 관점에서는 다음처럼 볼 수 있습니다.

- 비용을 더 써도 괜찮은데 성능이 그대로라면, 지금 단계에서 `large`로 바꿀 이유는 약합니다.
- 반대로 비용을 늘리지 않고도 `chunk_size`만 바꿔 더 좋은 결과를 얻었다면, 먼저 chunking을 손보는 것이 맞습니다.

### `chunk_overlap_50`

overlap을 줄이면 중복은 줄지만, 경계에 걸친 정보 손실이 커질 수 있습니다.

- `top1_accuracy`: `0.6667`
- `hit_rate_at_k`: `0.8333`
- `mrr`: `0.7222`

baseline 대비 분명한 하락이 있었고, 특히 positional encoding 관련 질의가 top-1에서 밀렸습니다. overlap은 단순한 중복 비용이 아니라, 문맥 보존 장치라는 점이 드러납니다.

### `chunk_size_500`

가장 많은 청크(`131개`)를 만들었지만, 성능은 가장 나빴습니다.

- 작은 청크는 특정 키워드 검색에는 유리할 수 있습니다.
- 그러나 문장과 문단의 연결이 끊겨서, retrieval ranking 품질은 오히려 나빠졌습니다.

특히 positional encoding 질의는 관련 청크를 top-3 안에 찾긴 했지만 rank 3에만 나타났고, scaled dot-product attention 질의도 top-1이 아닌 rank 2, 3에서만 관련성이 확인됐습니다.

사용자 경험으로 번역하면 이렇습니다.

- 시스템은 "어딘가에는 답을 갖고 있다"
- 하지만 "첫 답으로 잘 꺼내오지 못한다"

이런 설정은 내부적으로는 검색이 되는 것처럼 보여도, 최종 사용자에게는 답변 품질이 불안정한 시스템으로 느껴질 가능성이 큽니다.

## 3. 질의별로 본 핵심 패턴

### 가장 잘 맞는 질의들

아래 질의들은 거의 모든 설정에서 강했습니다.

- recurrence/convolution을 무엇으로 대체했는가
- multi-head attention은 어떻게 설명되는가
- scaled dot-product attention은 무엇을 계산하는가
- encoder/decoder stack 크기는 무엇인가

이 질의들의 공통점은 문서 안에 비교적 명시적인 표현이 있고, 특정 핵심 용어가 반복된다는 점입니다.

### 가장 어려운 질의

가장 어려운 질의는 아래였습니다.

- `Why is the Transformer good at modeling long-range dependencies?`

이 질의는 baseline, `chunk_size_500`, `chunk_overlap_50`, `retrieval_k_5`, `embedding_large`에서 모두 실패했고, `chunk_size_1500`에서만 성공했습니다.

이 결과는 두 가지를 시사합니다.

- 해당 답은 문서 안에 흩어진 개념을 묶어 읽어야 이해되는 유형일 수 있습니다.
- 의미적으로는 맞는 청크를 찾았더라도, 현재의 keyword-based relevance 판정 기준과 정확히 맞아떨어지지 않을 수 있습니다.

즉, 이 질의는 retrieval 자체의 난이도와 평가 기준의 한계를 동시에 보여줍니다.

실무적으로는 이런 질의가 더 중요할 수 있습니다. 사용자는 항상 문서 제목 그대로 질문하지 않고, 개념을 풀어서 묻기 때문입니다. 따라서 쉬운 factoid 질문만 잘 맞는 시스템보다, 이런 설명형 질문에서 덜 무너지는 시스템이 실제 만족도에 더 큰 영향을 줍니다.

### positional encoding 질의

`How does the model represent token order without recurrence?`도 chunking 전략에 민감했습니다.

- baseline: top-1 성공
- `chunk_size_500`: rank 3에서만 관련 청크 발견
- `chunk_size_1500`: rank 3에서만 관련 청크 발견
- `chunk_overlap_50`: rank 3에서만 관련 청크 발견
- `embedding_large`: top-1 성공

즉, 이 질의는 chunking과 embedding 양쪽의 영향을 모두 받는 케이스였습니다.

## 4. 시각화 해석

### `04_knowledge_base_overall_scores.png`

이 그래프는 설정별 `top1_accuracy`, `hit_rate_at_k`, `mrr`를 한 번에 비교합니다.

- `chunk_size_1500` 막대가 가장 안정적으로 높게 나와야 정상입니다.
- `chunk_size_500`는 특히 `top1_accuracy`와 `mrr`에서 가장 낮게 보일 것입니다.
- `embedding_large`와 `retrieval_k_5`는 기대보다 baseline과 큰 차이가 없다는 점이 눈에 띕니다.

### `04_knowledge_base_query_heatmap.png`

이 히트맵은 어떤 설정이 어떤 질의에 약한지를 가장 빠르게 보여줍니다.

- `long-range dependencies` 행에서 대부분의 설정이 낮은 값을 보일 것입니다.
- `chunk_size_1500`만 이 질의에서 상대적으로 높은 값을 보이면, 큰 청크가 분산된 문맥을 묶는 데 도움이 되었음을 의미합니다.

### `04_knowledge_base_chunk_tradeoffs.png`

이 그래프는 청크 수와 검색 성능의 관계를 보여줍니다.

- 청크 수가 많아질수록 항상 성능이 좋아지는 것은 아닙니다.
- 이번 실험에서는 `131개` 청크를 만든 `chunk_size_500`가 가장 비효율적이었습니다.
- 적당히 큰 청크가 오히려 더 좋은 retrieval 품질을 만들 수 있음을 확인할 수 있습니다.

## 5. 실무 관점 해석

이번 결과는 다음 순서로 튜닝하는 것이 합리적이라는 점을 보여줍니다.

1. 먼저 chunking을 조정합니다.
2. 그 다음 retrieval depth(`k`)를 조정합니다.
3. 마지막으로 embedding 모델 업그레이드를 검토합니다.

이유는 명확합니다.

- `chunk_size` 변화는 실제 메트릭 차이를 만들었습니다.
- `k`를 늘리는 것은 관련 문서를 더 많이 가져오지만, 첫 결과의 품질 개선으로 이어지지 않았습니다.
- `text-embedding-3-large`는 비용 증가 대비 이번 실험에서는 뚜렷한 품질 개선을 만들지 못했습니다.

여기서 중요한 것은 "어떤 제품을 만들고 있는가"에 따라 의사결정이 달라져야 한다는 점입니다.

### 사용자가 검색 결과를 직접 읽는 제품이라면

예: 사내 문서 검색, 리서치 도우미, 논문 검색

- `hit_rate_at_k`가 중요합니다.
- `k=5`로 늘리는 선택이 나쁘지 않을 수 있습니다.
- 다만 중복 문서 비율이 높아지면 결과 다양성이 떨어질 수 있으므로 reranking이나 deduplication이 필요합니다.

### 사용자가 한 번에 답변만 받는 챗봇이라면

예: 사내 Q&A 봇, 고객지원 봇, 튜터링 봇

- `top1_accuracy`와 `mrr`가 더 중요합니다.
- 이번 결과만 보면 `chunk_size=1500` 쪽이 더 적합합니다.
- `k`만 늘리는 것보다, 처음부터 더 좋은 1위 문서를 가져오게 만드는 것이 중요합니다.

### 비용에 민감한 시스템이라면

- `text-embedding-3-large`는 이번 실험에서 투자 대비 효과가 크지 않았습니다.
- 따라서 `text-embedding-3-small`을 유지하고 chunking을 먼저 튜닝하는 편이 합리적입니다.

### 답변 근거를 여러 개 묶어 요약하는 시스템이라면

- `relevant_count`가 너무 낮아도 안 좋습니다.
- 하지만 이번 실험처럼 `relevant_count`만 늘고 `mrr`가 그대로라면, 단순히 더 많이 가져오는 것만으로는 충분하지 않습니다.
- 이런 경우에는 `k` 확대보다 reranker 도입이 더 직접적인 개선책일 수 있습니다.

## 6. 다음 실험 제안

현재 평가는 `expected_keywords` 기반이므로, 실제 사용자 품질과 100% 동일하지는 않습니다. 다음 단계로는 아래가 유효합니다.

- keyword match 대신 수동 relevance 라벨셋 만들기
- top-k retrieved chunks를 LLM reranker에 통과시키기
- `chunk_size=1200`, `chunk_overlap=100` 같은 중간값 탐색
- query expansion 또는 HyDE 같은 질의 개선 기법 적용
- retrieval 점수 외에 최종 answer 정확도까지 함께 평가

## 7. 최종 요약

이번 실험의 가장 중요한 메시지는 다음 세 줄로 정리할 수 있습니다.

- 이 문서에서는 `chunk_size=1500`이 가장 좋았습니다.
- 작은 청크(`500`)는 문맥을 너무 잘게 쪼개서 retrieval 품질을 떨어뜨렸습니다.
- `embedding_large`나 `k=5`보다 chunking 전략이 더 큰 영향을 주었습니다.

한 문장으로 더 실무적으로 정리하면 다음과 같습니다.

- "돈을 더 쓰기 전에, 더 큰 모델을 쓰기 전에, 먼저 청크를 어떻게 자를지부터 다시 결정해야 한다."
