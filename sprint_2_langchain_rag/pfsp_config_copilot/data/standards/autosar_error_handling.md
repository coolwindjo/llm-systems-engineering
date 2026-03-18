# AUTOSAR Communication and Error Handling Notes

## Service communication expectations

AUTOSAR communication services should define a clear service identifier, signal or service class, timing behavior, and reaction strategy when communication deadlines are violated.
For cyclic publication, the sender should document the nominal update period and a timeout reaction that can be monitored by the receiver.

## Error handling and diagnostics

When a service is safety relevant, diagnostic reporting should describe how transmission failures, stale data, and invalid payload values are detected.
If communication is on-change, the design should still state how frozen values are recognized and escalated.
Diagnostic events should reference a fallback behavior such as degraded mode, safe default values, or controlled service disablement.

## Configuration guidance

Configuration reviews should verify that the service class matches the triggering mode:

- event or signal-style publication often maps to periodic or on-change broadcast behavior,
- method or request-response services should document whether repeated calls are allowed,
- one-shot services should state the trigger condition and post-trigger reset behavior.

Missing timing information or mismatched play type definitions should be treated as review findings before integration.
