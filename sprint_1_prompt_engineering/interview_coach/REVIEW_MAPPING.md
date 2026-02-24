# Interview Coach 리뷰 매핑 문서

이 문서는 `sprint_1_prompt_engineering/interview_coach` 프로젝트의
- 폴더 구조
- `05 Build an Interview Practice App.md` 요구사항 대비 구현 상태
- `pydantic` 사용 위치
를 리뷰 관점에서 빠르게 파악할 수 있게 정리한 문서입니다.

## 1) 전체 폴더 구조 (핵심 중심)

```text
interview_coach/
├─ app.py                          # Streamlit 앱 엔트리포인트
├─ cli.py                          # JD/프로필 추출 및 관리 CLI
├─ requirements.txt                # 의존성 (pydantic 포함)
├─ README.md                       # 프로젝트 개요/실행/매핑 설명
│
├─ components/                     # UI 레이어 (Streamlit)
│  ├─ app_runtime.py               # 세션 초기화, 모델/기법 선택, 패널 조립
│  ├─ sidebar.py                   # 사이드바 UI, 인터뷰어/프로필 관리
│  ├─ interview_session.py         # 인터뷰 시작(첫 질문 생성)
│  ├─ chat.py                      # 멀티턴 채팅, 피드백/힌트/모범답안
│  ├─ coding.py                    # 코딩 탭/패널 구성
│  └─ evaluation_dashboard.py      # 벤치마크 실행 + Judge 결과 시각화
│
├─ services/                       # 도메인 로직 레이어
│  ├─ interview_ops.py             # OpenAI 호출, 파싱, 모델 fallback, 프로필 생성
│  ├─ personas.py                  # 기법별 시스템 프롬프트 생성
│  ├─ profile_health.py            # 프로필 무결성 점검
│  └─ jd_keyword_catalog.py        # JD 키워드 카탈로그 처리
│
├─ utils/                          # 유틸리티
│  ├─ security.py                  # 입력 보안 가드
│  ├─ data_loader.py               # JSON 데이터 로딩
│  ├─ interviewer_store.py         # 인터뷰어 저장소 + 검증(Pydantic)
│  └─ model_temperature_constraints.json
│
├─ data/
│  ├─ interview_data.json          # 기본 인터뷰 데이터
│  ├─ interviewee_profile.json     # 후보자 프로필
│  ├─ profiles/                    # JD별 프로필 저장소
│  └─ interviewers/                # 인터뷰어 프로필 저장소
│
├─ prompts/
│  └─ system_prompts.json          # critique 관련 프롬프트 템플릿
│
├─ scripts/                        # 벤치마크/판정/분석 파이프라인
│  ├─ benchmark_suite.py
│  ├─ benchmark_suite_config.py    # 모델/기법/평가프롬프트 상수
│  ├─ benchmark_suite_judge.py     # LLM-as-a-Judge
│  ├─ benchmark_suite_analysis.py  # 결과 분석/차트
│  ├─ benchmark_suite_simulation.py
│  ├─ migrate_jd_keyword_catalog.py
│  └─ audit_interview_profiles.py
│
├─ tests/                          # 단위 테스트
├─ assets/readme/                  # README 스크린샷
└─ benchmark_logs/                 # 벤치마크 결과 산출물
```

## 2) `05 Build an Interview Practice App.md` 요구사항 매핑

## 필수 요구사항

| 요구사항 | 구현 상태 | 근거 파일 |
|---|---|---|
| 인터뷰 준비 도메인 정의/탐색 | 구현됨 (JD 프로필 기반으로 런타임 대상 포지션 변경 가능) | `components/app_runtime.py`, `utils/data_loader.py` |
| Streamlit/Next.js로 프론트엔드 구현 | 구현됨 (Streamlit 단일 앱 + 탭/사이드바) | `app.py`, `components/*` |
| OpenAI API Key 사용 | 구현됨 (`st.secrets` 또는 `OPENAI_API_KEY` 사용) | `components/app_runtime.py`, `components/chat.py`, `components/interview_session.py` |
| 지정 모델 사용 | 구현됨 (여러 OpenAI 모델 선택 + fallback) | `services/interview_ops.py` (`CHAT_CAPABLE_MODELS`, `create_chat_completion_with_fallback`) |
| 5개 이상 프롬프트 기법 | 구현됨 (`zero_shot`, `few_shot`, `chain_of_thought`, `persona_conditioning`, `knowledge_paucity`) | `components/app_runtime.py` (`PROMPT_TECHNIQUES`), `services/personas.py` |
| OpenAI 설정값 튜닝(최소 1개) | 구현됨 (`temperature` slider + 모델별 제약 clamp) | `components/app_runtime.py`, `services/interview_ops.py` (`_coerce_temperature_for_model`) |
| 보안 가드 최소 1개 | 구현됨 (입력 길이 제한 + 프롬프트 인젝션 키워드 차단) | `utils/security.py`, `components/chat.py` |

## 선택 과제 관점 (핵심)

| 선택 과제 항목 | 상태 | 근거 |
|---|---|---|
| LLM-as-a-Judge | 구현됨 | `scripts/benchmark_suite_judge.py`, `scripts/benchmark_suite_config.py`, `components/evaluation_dashboard.py` |
| 풀챗봇 형태(멀티턴) | 구현됨 | `components/chat.py`, `components/interview_session.py` |
| 인터뷰어 페르소나/다양화 | 구현됨 | `utils/interviewer_store.py`, `components/sidebar.py`, `services/personas.py` |
| 보안 고도화 | 부분 구현 | 현재는 `validate_input` 중심, 시스템 프롬프트 레벨 검증 확장은 여지 있음 |
| 비용 계산, 배포, 멀티-LLM 제공자 | 미구현/범위 밖 | 코드베이스 내 직접 구현 없음 |

## 3) Pydantic 사용 정리 (중요)

이 프로젝트에서 `pydantic`은 크게 3가지 역할로 사용됩니다.

1. **저장 데이터 스키마 검증 (Interviewer 프로필)**
2. **LLM 구조화 출력 파싱/검증 (JD/이력서/인터뷰어 추출)**
3. **UI 입력 임시 구조 정규화 (사이드바 form 파싱)**

### A. `utils/interviewer_store.py`

- `InterviewerProfile(BaseModel)`
  - 인터뷰어 저장 포맷의 표준 스키마
  - 필드: `name`, `background`, `is_generic_ai`, `role`, `expertise`, `potential_questions`, `critique_profile`
- `load_interviewer`/`_load_interviewer`에서 `InterviewerProfile.model_validate(raw)` 사용
  - JSON 파일 로드시 스키마 검증
  - 검증 실패(`ValidationError`) 시 `None` 처리하여 불량 파일 격리
- `save_interviewer`에서 `profile.model_dump()` 기반 직렬화
  - 저장 전에 리스트/불리언 정규화 수행

의미: 저장소 레벨에서 인터뷰어 데이터 품질을 보장합니다.

### B. `services/interview_ops.py`

- 구조화 추출 모델 정의
  - `ParsedIntervieweeProfile(BaseModel)`
  - `ParsedInterviewerProfile(BaseModel)`
  - `JDExtraction(BaseModel)`
- OpenAI Structured Output 파싱
  - `client.beta.chat.completions.parse(..., response_format=모델)` 패턴 사용
  - 예: `parse_interviewee_profile`, `parse_interviewer_background`, `extract_jd_fields`
- fallback 경로에서도 Pydantic 재검증
  - LLM이 일반 텍스트/JSON 문자열을 반환한 경우 `model_validate_json(...)` 사용
  - 필요시 `model_validate(parsed_dict)`로 최종 검증
- `ValidationError`를 예외 핸들링에 포함
  - 모델 접근 오류/파싱 실패 시 모델 fallback 흐름 유지

의미: LLM 응답을 "타입 보장된 구조체"로 강제해 downstream 로직 안정성을 높입니다.

### C. `components/sidebar.py`

- `_ExperienceDraft(BaseModel)`
  - 사이드바의 인터뷰어 경험 텍스트 블록을 구조화(`background`, `expertise`, `potential_questions`)
- `_parse_experience_block`에서 `model_validate(payload)` 사용
  - 자유형 텍스트 입력을 저장 가능한 내부 구조로 정규화

의미: UI 입력 품질을 최소한의 스키마로 고정해 저장 오류를 줄입니다.

### D. 의존성 선언

- `requirements.txt`에 `pydantic` 포함.

## 4) 리뷰 시 확인하면 좋은 포인트

- `pydantic` 검증 실패 시 사용자 피드백이 충분한지(UI 메시지 명확성)
- `ValidationError` 처리 후 fallback 경로가 의도대로 동작하는지(특히 모델 권한 오류와 구분)
- `sidebar.py` 자유형 입력 파싱 규칙이 실제 사용자 입력 패턴과 맞는지
- 벤치마크 Judge 결과 JSON 파싱 실패 시(현재 fail-safe 1점 처리) 운영 정책이 적절한지

