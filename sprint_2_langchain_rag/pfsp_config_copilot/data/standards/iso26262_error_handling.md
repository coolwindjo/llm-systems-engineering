# ISO 26262 Inspired Safety Guidance for Communication Services

## Safety-oriented requirement quality

Safety-related software requirements should be unambiguous, internally consistent, and verifiable.
When a requirement mixes conflicting timing modes, for example both cyclic and on-change in the same statement, the conflict should be resolved before implementation.

## Error detection and safe state

If a communication service contributes to a safety goal, the system should define:

- what constitutes a communication fault,
- how quickly the fault must be detected,
- which diagnostic or supervision mechanism observes it,
- and what safe-state or degraded reaction is expected.

## Timing and validation

A frequency value should be explicit enough to support verification, such as `10 ms` or `100 Hz`.
If the frequency is missing, a reviewer should ask for the expected update cadence or the condition that triggers transmission.
If the service identifier is missing or ambiguous, configuration generation should stop until the identifier is clarified.

## Recommendation style

Good validation output should distinguish:

1. extracted parameters that are sufficiently specified,
2. unresolved fields that require user clarification,
3. safety and diagnostics recommendations tied to the requirement context.
