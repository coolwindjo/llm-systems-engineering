# Tool Calling Evaluation Lab 해설서 🚀

이 문서는 [sprint2_part4_lab2_tool_calling_eval.ipynb](file:///workspace/sprint_2_langchain_rag/notebooks/sprint2_part4_lab2_tool_calling_eval.ipynb) 노트북을 통해 Tool Calling 에이전트의 성능을 평가(Evaluation)하는 핵심 개념과 파이프라인 구축 방법을 이해하도록 돕기 위해 작성되었습니다.

---

## 1. Tool Calling 평가는 왜 필요한가요? 🤔

LLM 에이전트가 완벽하게 동작하려면, 사용자의 질문에 맞게 **"어떤 도구(Tool)를 쓸지"** 그리고 **"어떤 인자(Parameters)를 넘길지"**를 정확히 결정해야 합니다.
이 중 하나라도 어긋나면 엉뚱한 결과를 낳게 됩니다. (예: 급여 비교를 해야 하는데 채용 공고를 검색함, 혹은 베를린 대신 런던을 검색함 등)

따라서 우리는 다음과 같은 **두 가지 상호 보완적인 평가 방법**을 배웁니다.
1.  **Deterministic(결정적) 평가**: 코드(문자열) 비교를 통해 정답과 정확히 일치하는지 빠르고 저렴하게 채점합니다.
2.  **LLM-as-a-judge (LLM 심사위원)**: RAGAs 등의 프레임워크를 사용하여, 예측했던 정답과 완벽히 일치하지 않아도 "상황상 합리적인 선택이었는지" 정성적/맥락적으로 평가합니다.

---

## 2. 노트북 주요 섹션 가이드 📝

### A. 평가의 3단계 (Three Levels of Correctness)
Tool Calling이 실패할 수 있는 세 가지 지점을 확인하고 각기 다른 평가를 수행합니다.
*   **Tool Selection (도구 선택)**: 올바른 도구를 골랐는가? (Deterministic)
*   **Parameter Extraction (인자 추출)**: 도구에 필요한 파라미터를 정확히 뽑아냈는가? (Deterministic)
*   **Answer Quality (답변 품질)**: 최종 응답(자연어)이 상황에 알맞고 훌륭한가? (LLM-as-a-judge)
*(본 노트북은 앞의 2개 항목과 도구 선택의 타당성에 대한 LLM 평가에 집중합니다.)*

### B. 에이전트 및 도구 정의 (`@tool`)
평가를 진행할 "커리어 컨설턴트" 에이전트를 만듭니다.
*   `search_jobs`, `compare_salaries`, `analyze_resume` 3개의 도구를 파이썬 데코레이터(`@tool`)로 손쉽게 정의하고, LangChain의 `create_agent`로 묶어서 에이전트를 구성합니다.

### C. 테스트 데이터셋 설계 (Test Dataset Design)
평가에서 가장 중요한 것은 **좋은 시험 문제(Dataset)**를 만드는 것입니다. 노트북에서는 14개의 질문을 난이도별로 구성했습니다.
*   **Easy**: 의도가 명확하고 도구 매칭이 직관적임 (예: "베를린의 데이터 사이언티스트 평균 급여는?")
*   **Medium**: 간접적인 표현이나 헷갈리는 키워드 포함 (예: "제품 관리자가 얼마나 버는지 '검색(search)'해 줄래?" -> '검색'이라는 단어 때문에 `search_jobs`로 착각하기 쉽지만 `compare_salaries`가 정답입니다.)
*   **Hard**: 의도가 섞여 있거나 여러 도구 중 모호한 상황
*   **Edge cases**: 도구를 쓰면 안 되는 일상적인 대화 (예: "농담해줘")

---

## 3. 평가 방식 상세 분석 (Evaluation Methods) 💡

### 방법 1. Deterministic Evaluation (결정적 평가)
*   **방식**: 우리가 미리 정의한 정답(`expected_tool`, `expected_params`)과 LLM이 실제 선택한(`actual_tool`, `actual_params`) 값을 파이썬의 `==` 연산자로 단순 비교합니다.
*   **코드 핵심**: 
    ```python
    tool_correct = actual_tool == case["expected_tool"]
    params_correct = actual_params == case["expected_params"]
    ```
*   **장단점**: 매우 빠르고 비용이 들지 않지만(Fast, Free), 융통성이 없습니다. (예: 예상치 못한 도구를 썼지만 논리적으로는 맞는 경우 무조건 FAIL 처리됨)

### 방법 2. LLM-as-a-judge 평가 (`DiscreteMetric`)
*   **방식**: RAGAs 라이브러리의 `DiscreteMetric`을 사용하여, 또 다른 똑똑한 LLM(심사위원)에게 "이 에이전트가 선택한 도구가 상황에 비추어 볼 때 적절했는가?"를 묻습니다.
*   **코드 핵심**:
    프롬프트에 사용자의 질문, 예상 도구, 실제 선택 도구를 모두 넣어주고 심사위원 LLM이 `correct` 또는 `incorrect` 중 하나를 고르도록 합니다.
    ```python
    score = metric.score(
        llm=llm,
        user_query=r["query"],
        expected_tool=str(r["expected_tool"]),
        actual_tool=str(r["actual_tool"])
    )
    # score.value (correct/incorrect)와 score.reason (평가 이유) 반환
    ```
*   **장단점**: 에이전트의 선택이 왜 합리적이었는지, 혹은 왜 틀렸는지에 대한 **이유(`reason`)**를 상세히 알려줍니다. 단, 평가 비용과 시간이 발생합니다.

### 💡 핵심: 두 방식의 비교 (Disagreements)
노트북 후반부에서 두 방식의 결과를 비교(`!!` 표시)합니다. 
가장 흥미로운 부분은 **"Deterministic은 FAIL을 받았는데, LLM Judge는 PASS(correct)를 준 상황"**입니다.
*   단순 문자열 비교로는 오답(기대한 함수명과 다름)이지만, 실제 평가자(심사위원 LLM)가 볼 때는 해당 도구를 사용해 답변을 시도한 것도 "충분히 말이 되는 합리적인 대체 전략"일 때 주로 발생합니다.
*   이를 통해 우리는 엄격한 코드 테스트가 놓치는 유연하고 복잡한 언어적 맥락을 LLM 평가가 어떻게 훌륭하게 보완하는지 알 수 있습니다.

---

## 4. 학습 팁 🌟

1.  **점진적 도입 권장**: 처음에는 Deterministic(결정적) 방식으로 빠르게 대량의 테스트(기본 유효성 검사 등)를 한 뒤 문제점을 먼저 파악하고, 점차 고도화된 LLM Judge를 도입해 깊게 분석하는 흐름으로 평가 파이프라인을 구축하는 것이 정석입니다.
2.  **데이터셋의 중요성**: 평가 파이프라인을 아무리 잘 만들어도 평가 문제(Dataset)가 단순하고 치우쳐 있으면 에이전트의 성능을 과대평가하게 됩니다. 노트북의 예시처럼 '도구를 쓰면 안 되는 경우(Edge cases)'나 우회적인 표현(Medium/Hard)을 반드시 포함하세요.
3.  **작은 모델(Router)의 가능성**: 노트북 말미 부분에 언급되듯, 단순한 도구 선택 작업은 무겁고 비싼 최신 대형 모델보다 가벼운 모델(`gpt-3.5-turbo`나 전용 라우터 모델)이 오히려 더 지시를 잘 따르고 오버 엔지니어링을 피할 수 있습니다. 목적에 맞는 모델 선택의 인사이트를 얻어보세요.

즐거운 학습 되시길 바랍니다! 궁금한 점이 있으면 언제든 물어보세요.
