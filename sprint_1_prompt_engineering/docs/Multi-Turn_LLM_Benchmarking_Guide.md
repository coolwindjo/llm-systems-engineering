# Multi-Turn LLM-as-a-Judge Benchmarking Guide

이 문서는 면접 연습 앱(Interview Coach)의 멀티턴 대화 성능을 객관적으로 평가하고 비교하기 위한 벤치마크 시스템 구현 가이드입니다.

## 1. 멀티턴 평가 전략: User Proxy Simulator
멀티턴 대화에서는 매번 사용자의 입력이 달라지기 때문에 '고정된 입력'에 대한 성능 비교가 어렵습니다. 이를 해결하기 위해 **User Proxy(고정된 시뮬레이터)** 방식을 사용합니다.

- **방법**: 면접관 LLM이 어떤 답변을 하더라도, 사용자는 미리 정해진 3~5단계의 '시나리오 답변'을 순차적으로 내놓게 합니다.
- **장점**: 모든 시스템 프롬프트 버전이 **동일한 후보자 답변**을 마주하게 되어, 면접관의 '질문 품질'과 '피드백 깊이'를 1:1로 비교할 수 있습니다.

## 2. 벤치마크 러너 구현 (`benchmark_runner.py`)

```python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 1. 비교 테스트할 시스템 프롬프트 버전들
PROMPTS_TO_TEST = {
    "v1_basic": "You are a helpful interview coach.",
    "v2_cot": "You are an expert interview coach. Use Chain-of-Thought reasoning before asking questions.",
    "v3_expert": "You are a senior ADAS engineer. Ask deep technical questions including ASIL-B and ASPICE CL3."
}

# 2. 고정된 사용자 입력 (User Proxy)
USER_PROXY_TURNS = [
    "Hi, I'm ready. I have 10 years of experience in ADAS perception.",
    "I usually use Kalman filters for object tracking and CNNs for detection.",
    "That's all. Can you give me some feedback on my answers?"
]

# 3. 판사 루브릭 (Judge Rubric)
JUDGE_SYSTEM_PROMPT = """
You are an expert LLM evaluator. Score the following interview transcript (1-10) on:
1. Technical Depth: Does the coach ask deep, job-relevant questions?
2. Logical Flow: Is the transition between turns natural?
3. Feedback Quality: Is the final feedback actionable?
Return JSON only: {"tech_score": int, "flow_score": int, "feedback_score": int, "reasoning": "string"}
"""

def run_simulation(name, system_content):
    messages = [{"role": "system", "content": system_content}]
    for user_input in USER_PROXY_TURNS:
        messages.append({"role": "user", "content": user_input})
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        messages.append({"role": "assistant", "content": resp.choices[0].message.content})
    return messages

def evaluate(transcript):
    formatted = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in transcript if m['role'] != 'system'])
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": formatted}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

def generate_radar_chart(df):
    categories = ['tech_score', 'flow_score', 'feedback_score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, row in df.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['version'])
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig("benchmark_radar.png")

def main():
    results = []
    for name, content in PROMPTS_TO_TEST.items():
        transcript = run_simulation(name, content)
        eval_result = evaluate(transcript)
        eval_result['version'] = name
        results.append(eval_result)
    
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("benchmark_results.csv", index=False)
    generate_radar_chart(df)

if __name__ == "__main__":
    main()
```

## 3. 성능 향상을 위한 Advanced Probing 기술
'Technical Depth' 점수를 높이기 위해 시스템 프롬프트에 추가할 3가지 심층 질문 기법입니다.

1.  **Safety-Critical Constraints (ASIL-B/D)**:
    - "후보자가 기술적 해결책을 제시하면, 반드시 '그 설계가 ISO 26262 ASIL-B 표준을 충족해야 한다면 어떤 Redundancy를 추가할 것인가?'라고 되물으세요."
2.  **Quantitative Trade-off Analysis**:
    - "단순히 설명하게 두지 말고, '그 방식의 Latency와 Throughput 간의 Trade-off는 무엇인가?'와 같은 수치적/구조적 한계를 질문하세요."
3.  **Failure Mode Deep-Dive**:
    - "제시된 시스템의 특정 의존성(Dependency)이 실패했을 때의 'Cascading Effect'와 'Graceful Degradation' 전략을 요구하세요."

## 4. 최종 분석 리포트 템플릿
벤치마크 결과를 요약할 때 다음 형식을 사용하세요.

### 📊 Benchmarking Final Analysis Report
- **Best for Technical Depth**: `[v3_expert]` (Score: 9/10)
  - *Reasoning*: 구체적인 안전 표준(ASIL)을 언급하며 후보자의 한계를 시험함.
- **Best for Logical Flow**: `[v2_cot]` (Score: 8/10)
  - *Reasoning*: 단계별 추론을 통해 대화의 맥락을 가장 잘 유지함.
- **Conclusion**: 시니어 엔지니어 면접용으로는 `v3`를, 일반 코칭용으로는 `v2`를 채택하는 것이 권장됨.
