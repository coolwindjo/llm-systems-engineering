# Function Calling Lab 해설서 🚀

이 문서는 [sprint2_part4_lab1_function_calling_lab.ipynb](file:///workspace/sprint_2_langchain_rag/notebooks/sprint2_part4_function_calling_lab.ipynb) 노트북을 통해 Function Calling의 핵심 개념을 파악하고 실습을 성공적으로 마칠 수 있도록 돕기 위해 작성되었습니다.

---

## 1. Function Calling이란? 🤔

Function Calling은 LLM이 직접 외부 API를 호출하거나 파이썬 코드를 실행하는 것이 아닙니다. 대신, 사용자의 질문을 이해하고 **"내가 직접 답하기보다 이 함수를 이런 인자로 실행하면 좋겠어"**라고 **제안(Tool Call Proposal)**하는 기능입니다.

### 핵심 워크플로우 (Ping-Pong)
1.  **사용자 질문**: LLM에게 도구 목록과 함께 질문 전달.
2.  **LLM 응답**: `tool_calls`를 포함한 응답 (함수 이름과 인자가 들어있음).
3.  **Local 실행**: 개발자(코드)가 LLM이 제안한 인자로 실제 함수를 실행.
4.  **결과 전달**: 함수 결과를 LLM에게 다시 전달.
5.  **최종 응답**: LLM이 함수 결과를 바탕으로 자연스러운 문장으로 답변.

---

## 2. 노트북 주요 섹션 가이드 📝

### A. 도구 정의 (`tools` List)
노트북에서는 JSON Schema 형식을 사용하여 함수를 정의합니다.
*   **Description**: 모델이 언제 이 함수를 써야 할지 판단하는 근거가 되므로 상세히 적어야 합니다.
*   **`strict: True`**: 인자 형식을 엄격하게 제한하여 모델이 잘못된 타입의 데이터를 생성하지 않도록 돕습니다.
*   **`required`**: 필수 인자를 명시합니다.

### B. 실행 로직 (`execute_tool` 함수)
LLM이 제안한 도구 호출을 실제 파이썬 함수와 매핑하는 과정입니다. 
*   `tool_registry` 딕셔너리를 사용하여 문자열 이름(`"calculator_add"`)을 실제 함수 객체(`calculator_add`)와 연결하는 방식이 권장됩니다.

---

## 3. 실습 문제 (Exercise) 별 힌트 💡

### Exercise 2: 함수 추가 및 강제 호출 (`tool_choice`)
모델이 여러 도구 중 무엇을 쓸지 고민하게 두지 않고, **특정 도구를 무조건 쓰도록 강제**하는 설정입니다.

*   **설정 방법**: `tool_choice={"type": "function", "function": {"name": "함수이름"}}`
*   **왜 필요한가요? (Use Cases)**:
    1.  **데이터 구조화(Extraction)**: 사용자의 난잡한 텍스트를 특정 JSON 형식으로 뽑아내고 싶을 때, "추출용 함수"를 강제로 쓰게 만들면 모델이 헛소리를 하지 않고 정확히 JSON만 생성합니다.
    2.  **엄격한 워크플로우**: 예를 들어 "예약 시스템"에서 지금 단계는 무조건 '날짜 선택'이어야 한다면, 모델이 딴청 피우지 못하게 `select_date` 함수만 쓰도록 고정할 수 있습니다.
    3.  **디버깅**: 특정 함수가 인자를 잘 생성하는지 테스트하고 싶을 때 유용합니다.

### Exercise 3: 외부 API 연동 (`requests`)
노트북에서 실제 외부 API를 호출하는 부분은 **`get_chuck_norris_joke`** 함수 내부입니다.

*   **코드 위치**: 노트북 중간의 `def get_chuck_norris_joke(category: str):` 정의 부분.
*   **외부 API 호출 핵심 코드**:
    ```python
    url = "https://api.chucknorris.io/jokes/random"
    response = requests.get(url, params={"category": category}, timeout=8)
    payload = response.json()
    ```
*   **설명**: 
    1.  `url`: 호출할 서버의 주소입니다.
    2.  `requests.get()`: 파이썬의 `requests` 라이브러리를 사용하여 해당 URL에 데이터를 요청(Request)합니다.
    3.  `response.json()`: 서버로부터 받은 응답(Response)을 파이썬이 이해할 수 있는 딕셔너리(JSON) 형태로 변환합니다.
    
모델은 이 함수를 실행한 **결과물(payload)**을 보고 사용자에게 농담을 전달하게 됩니다.

### Exercise 4: 에러 핸들링 및 일반 응답 (`tool_calls` 체크)
이 연습 문제는 앞서 노트북의 **"Parse response" 셀에서 보았던 분기 처리 로직 중 '일반 대화 응답' 부분**이 어떻게 동작하는지 명시적으로 보여주는 예시입니다.

*   **Parse response의 핵심 로직 연관성**:
    노트북의 "Parse response" 셀에서는 모델의 응답에 따라 다음과 같은 분기 처리를 수행했습니다.
    ```python
    assistant_message = completion.choices[0].message
    
    if not assistant_message.tool_calls:
        # ▶ 이 연습 문제(Exercise 4)가 바로 이 분기(일반 대화)를 타는 상황을 가정한 것입니다.
        print(assistant_message.content)
    else:
        # 도구가 하나라도 있는 경우 (기존 루프 실행)
        for tool_call in assistant_message.tool_calls:
            # execute_tool 로직 수행...
    ```
*   **상세 설명**:
    1.  **일반 대화 유도**: 이 연습 문제에서는 `tool_choice="none"` 옵션을 주거나 일반적인 인사말("Hello!")을 건네어 LLM이 함수를 부르지 **않도록** 유도합니다.
    2.  **동작 원리**: LLM이 도구를 달라고 하지 않으면 `assistant_message.tool_calls` 속성은 비어있게 됩니다. 실제 어플리케이션 환경에서는 위 Parse response와 동일한 로직을 거쳐서, 함수를 실행하지 않고 곧바로 `message.content` 속성(인사말에 대한 답변)을 출력하게 됩니다.
    3.  **에러 핸들링 (안전망)**: 만약에 함수를 호출했을지라도, 내가 알지 못하는 함수 이름(`Unknown tool`)을 가져왔거나 예상과 다른 인자를 가져왔을 경우에 프로그램이 뻗지 않도록 `execute_tool` 함수 내부에 `try-except` 블록을 두고 예외 처리하게 됩니다.
    
결과적으로, 이 "에러 핸들링 및 일반 응답" 로직은 챗봇이 **"똑똑한 비서(도구 사용)"**와 **"친절한 대화 상대(일반 답변)"** 역할을 모두 안전하게 수행할 수 있게 해주는 필수 안전장치입니다.

### Exercise 5: 재사용 가능한 도구 호출 실행 루프 (Execution Loop) 구축
*   **실제 어플리케이션 환경(Production-ready)**처럼 사용자 프롬프트를 받아서 도구를 호출하고 그 결과를 바탕으로 최종 답변을 반환하는 과정을 하나의 `run_conversation` 함수로 만들어 재사용성을 높인 예제입니다.
*   **핵심 과정**:
    1.  **첫 번째 모델 호출**: 사용자의 질문(`user_prompt`)을 받아 LLM에게 전달합니다.
    2.  **분기 처리 (`if not response_message.tool_calls`)**: LLM이 도구 사용이 필요 없다고 판단하면 별도의 코드 실행 없이 그대로 시스템 답변을 반환하고 종료합니다. (Exercise 4 부분의 응용입니다)
    3.  **도구 실행 및 메시지 추가 (`messages.append`)**: LLM이 제안한 도구들(`tool_calls`)을 `execute_tool`로 차례대로 실행한 후, 그 결과(`JSON`)를 `role: "tool"` 개체 형식으로 기존 대화 이력(`messages`)에 추가합니다. 
    4.  **두 번째 모델 호출**: 도구 실행 결과가 추가된 전체 대화 기록을 LLM에게 다시 전달하여 최종적인 자연어 답변을 완성합니다.
*   앞서 낱개로 흩어져 있던 LLM 제안 호출 -> Parse & Execute -> 결과 취합 후 LLM 최종 호출 과정이 하나의 흐름으로 통합된 형태라고 이해하시면 됩니다.

**💡 참고: `model_dump()` 함수란 무엇인가요?**
코드 중에 `messages.append(response_message.model_dump())` 라는 부분이 있습니다.
*   **역할**: OpenAI 최신 파이썬 라이브러리는 응답 결과를 단순한 딕셔너리(사전) 형태가 아니라, `Pydantic`이라는 라이브러리를 기반으로 한 **객체(Object)** 형태로 반환합니다.
*   **이유**: 우리가 이전에 보낸 메시지 목록(`messages = [{"role": "user", ...}]`)은 파이썬 딕셔너리 형태입니다. 따라서 응답으로 받은 객체를 이 리스트에 안전하게 다시 추가하려면, **객체를 다시 딕셔너리 형태로 변환**해주어야 합니다.
*   **요약**: `model_dump()`는 OpenAI가 준 복잡한 응답 객체를 우리가 다루기 쉬운 기본 **파이썬 딕셔너리 형태로 덤프(변환/추출)해주는 내장 함수**입니다.

### Exercise 6: RAG + Function Calling 조합
이 연습 문제는 **RAG(Retrieval-Augmented Generation, 검색 증강 생성)** 기술과 **Function Calling** 기술을 어떻게 하나로 합쳐서 똑똑한 챗봇을 만들 수 있는지 보여주는 핵심 예제입니다.

*   **동작 원리 (코드 상세 분석)**:
    1.  **가짜 지식 베이스 (`knowledge_base`)**: 실제 벡터 DB(Chroma, Pinecone 등) 대신, 데모를 위해 리스트 형태로 문서(딕셔너리)들을 저장한 메모리 공간입니다.
    2.  **검색 알고리즘 (`retrieve_context`)**: 사용자의 질문(`query`) 단어들과 지식 베이스 문서의 단어들이 얼마나 일치하는지(교집합)를 단순 계산하여 관련 문서를 찾아내는 가짜 검색 도구(Retriever)입니다.
    3.  **RAG용 도구 목록 병합 (`rag_tools`, `rag_tool_registry`)**: 앞서 만든 기존 도구들(`tools`: 계산기, 농담 API)에 방금 만든 검색 도구(`retrieve_context`)를 더하여 도구 목록을 확장합니다. 이제 챗봇은 일반 도구와 검색 도구를 모두 쓸 수 있습니다.
    4.  **사용자 프롬프트 (`rag_messages`)**: *"RAG와 함수 호출 동작 방식을 설명하고, 농담도 하나 해줘"* 라고 복합적인 질문을 던집니다.
    5.  **첫 번째 모델 호출 (`rag_completion` & 병렬 호출)**: LLM이 사용자의 복합 질문을 보고, **"아, 이 질문에 답하려면 `retrieve_context` 함수로 문서를 검색하고, `get_chuck_norris_joke` 함수로 농담을 가져와야겠구나"**라고 판단하여 **두 개의 도구를 동시에(병렬로) 제안**합니다.
    6.  **도구 실행 및 결과 반영 (`execute_rag_tool`, `rag_follow_up_messages`)**: 제안받은 모든 도구를 `for tool_call in rag_assistant_message.tool_calls:` 루프를 돌며 실행합니다. 지식 베이스 내용과 외부 API 농담 결과가 각각 파이썬 딕셔너리로 추출되고, 이를 `role: "tool"` 포맷으로 묶어 기존 대화 이력(`rag_follow_up_messages`)에 담아줍니다.
    7.  **두 번째 모델 호출 (`rag_final`)**: 챗봇은 방금 가져온 문서 내용(RAG 지식)과 농담(API 결과)을 취합해서 최종적으로 자연스러운 하나의 문단(답변)으로 완성하여 사용자에게 제공합니다.

**💡 참고: `**tool_registry`에서 `**`는 무슨 뜻인가요?**
코드 중에 `rag_tool_registry = { **tool_registry, "retrieve_context": ... }` 라는 부분이 있습니다. 
*   **의미 (Dictionary Unpacking)**: 파이썬에서 `**`는 딕셔너리 안에 있는 모든 내용을 **"풀어서(Unpack)"** 다른 딕셔너리에 합칠 때 사용합니다.
*   **왜 사용하나요?**: 기존에 만들어둔 계산기와 농담 도구들이 담긴 `tool_registry`를 그대로 복사해오면서, 거기에 새로운 `retrieve_context` 함수만 **추가(Merge)**하기 위해 사용합니다. 
*   **쉽게 말하면?**: "기존 딕셔너리에 들어있는 거 다 여기다 넣고, 이것도 하나 더 추가해줘!"라는 뜻입니다.

*   **핵심 의의**: 이 과정을 통해 LLM이 **여러 개의 서로 다른 도구(내부 DB 검색 + 외부 API 호출)를 상황에 맞게 조합**하여 복잡한 지시를 수행할 수 있음을 체감할 수 있습니다.

### 부록: [Exercise 6 vs Exercise 7] 순수 파이썬 vs LangChain 구현 비교
Exercise 6은 내부 구조를 이해하기 위해 OpenAI API와 순수 파이썬만으로 구현한 반면, Exercise 7은 **LangChain**을 사용하여 동일한 기능을 훨씬 간결하게 구현합니다.

| 구현 절차 (Procedure) | 순수 파이썬 (Exercise 6) | LangChain (Exercise 7) | 특이사항 비교 |
| :--- | :--- | :--- | :--- |
| **1. 도구(함수) 정의** | 복잡한 딕셔너리로 **JSON Schema** 명세 직접 작성 (`type`, `properties`, `required` 등 지정) | 파이썬 함수 위에 **`@tool` 데코레이터**만 추가 | LangChain은 함수 시그니처와 Docstring을 읽어 자동으로 JSON Schema를 생성하므로 훨씬 편리함. |
| **2. 도구 명부(Registry) 설정** | 함수 실행을 위해 함수 이름을 파이썬 함수로 매핑하는 딕셔너리(`rag_tool_registry`) 직접 생성 | LangChain 도구 객체를 리스트로 묶어 전달하면 됨 | LangChain은 내부적으로 도구 매핑과 관리를 처리해줌. |
| **3. LLM에 도구 묶기 (Binding)** | 모델 호출 시마다 `tools=rag_tools`로 리스트를 매번 전달 | `ChatOpenAI().bind_tools([tools])` 메서드를 사용해 객체화 | LangChain은 설정된 도구가 고정된(Bound) 새로운 LLM 객체를 반환하여 재사용성을 높임. |
| **4. 모델 호출 (실행)** | `client.chat.completions.create(messages=...)` | `llm_with_tools.invoke(messages)` | `invoke()` 하나로 호출 방식이 통일되고 간결해짐. |
| **5. 도구 결과 메세지 조립** | `{"role": "tool", "tool_call_id": ..., "content": ...}` 형태의 **딕셔너리 직접 생성** 및 Append | LangChain 전용 클래스인 **`ToolMessage(...)` 객체 생성** 및 Append | LangChain은 메세지 타입(`SystemMessage`, `HumanMessage`, `ToolMessage`)을 객체 지향적으로 관리하여 실수를 줄임. |

---

## 4. 학습 팁 🌟

1.  **API 비용 주의**: 잦은 호출은 비용을 발생시키므로, 코드가 논리적으로 완성된 후 실제 호출을 진행하세요.
2.  **보안**: `OPENAI_API_KEY`와 같은 비밀키는 `.env` 파일이나 Colab의 `Secrets` 탭을 이용해 안전하게 관리하세요.
3.  **JSON Schema**: 처음에는 복잡해 보일 수 있지만, GPT가 원하는 규격을 맞추기 위한 약속입니다. `parameters`의 `type`과 `properties` 구조에 익숙해지는 것이 중요합니다.

즐거운 학습 되시길 바랍니다! 궁금한 점이 있으면 언제든 물어보세요.
