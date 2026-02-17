# 구조적 출력의 심층 분석과 활용 방법

이 문서는 `05_practice_structured_outputs_deep_dive.py`가 구조적 출력(Structured Outputs)의 심화 기법을 어떻게 실습하는지 정리한 설명서입니다.  
핵심은 다음 3가지입니다.

1. 선택적 필드(Optional Fields)
2. 열거형(Enum)을 활용한 환각 방지
3. 2단계 접근법(Two-Phased Approach)

## 1) Pydantic 기반 스키마 정의

문서에서는 원시 JSON Schema를 직접 관리하기보다 `Pydantic`으로 타입/스키마를 선언하는 방식을 권장합니다.  
코드에서는 `BaseModel`을 상속한 아래 클래스들이 해당 역할을 수행합니다.

- `Webpage`
- `RerankingResult`
- `TwoPhaseReasoning`

이 방식은 Python 타입 힌트를 구조적 출력 스키마로 자연스럽게 연결해, 코드 가독성과 유지보수성을 높입니다.

## 2) Optional 필드로 유연성 확보

구조적 출력은 스키마 형태를 강하게 보장하지만, 실제 입력 데이터는 항상 완전하지 않을 수 있습니다.  
이때 `Optional` 필드를 사용하면 값이 없는 경우 `null`을 허용하여 유연성을 확보할 수 있습니다.

### 코드 연결 지점

- `Webpage` 클래스
- `optional_fields_demo()` 함수

예시:

```python
class Webpage(BaseModel):
    links: Optional[list[str]] = Field(None, ...)
    images: Optional[list[str]] = Field(None, ...)
```

HTML에 `<a>`나 `<img>`가 없을 때 억지 값 생성 대신 `null`을 허용하므로, 불필요한 환각을 줄이고 실제 데이터 상태를 정확히 반영할 수 있습니다.

## 3) Enum으로 환각 감소

자유 문자열(예: 제품 ID)을 직접 생성하게 하면 모델이 존재하지 않는 값을 만들 가능성이 있습니다.  
이를 줄이기 위해 출력 후보를 고정 집합(Enum)으로 제한합니다.

### 코드 연결 지점

- `Rank(IntEnum)` 클래스
- `enum_reranking_demo()` 함수

예시:

```python
class Rank(IntEnum):
    RANK_1 = 1
    ...
    RANK_5 = 5
```

모델이 반환할 수 있는 값이 `1~5`로 제한되므로, 스키마 범위를 벗어나는 출력 위험이 크게 줄어듭니다.

## 4) 2단계 접근법(Two-Phased Approach)

복잡한 판단 문제에서 한 번에 구조화 출력만 강제하면, 추론 품질이 떨어질 수 있습니다.  
이를 보완하기 위해 **추론 단계**와 **구조화 단계**를 분리합니다.

### 코드 연결 지점

- `two_phase_demo()` 함수

### 단계 구성

1. **Phase 1 (Reasoning)**  
   `client.responses.create`로 자유 텍스트 추론을 먼저 생성합니다.
2. **Phase 2 (Structuring)**  
   1단계 결과 텍스트를 입력으로 `client.responses.parse`를 호출하여 `TwoPhaseReasoning` 스키마로 정규화합니다.

이 패턴은 자연어 추론의 유연성과 구조적 출력의 결정론적 안정성을 함께 확보하는 실무형 접근입니다.

## 결론

`05_practice_structured_outputs_deep_dive.py`는 다음을 실전적으로 보여줍니다.

- 스키마 안정성: Pydantic 기반 구조화
- 유연한 결측 처리: Optional
- 환각 억제: Enum 제한
- 복잡 과제 대응: 2단계 파이프라인

즉, 구조적 출력은 단순 JSON 형식화를 넘어, LLM의 비정형 추론 결과를 신뢰 가능한 소프트웨어 인터페이스로 연결하는 핵심 기법입니다.

