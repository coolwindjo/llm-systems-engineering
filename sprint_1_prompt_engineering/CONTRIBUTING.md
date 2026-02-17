# Contributing to SensorStreamKit

Thank you for your interest in contributing to SensorStreamKit!

## Commit Message Convention

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>: <subject>

[optional body]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, whitespace, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (build scripts, dependencies, etc.) |
| `perf` | Performance improvements |

### Subject Rules

- Use imperative mood ("add feature" not "added feature")
- Don't capitalize the first letter
- No period at the end
- Keep it under 50 characters

### Body Rules (Optional)

- **Why**: Describe the motivation for the change
- **How**: Describe details or side effects
- Use only when necessary; omit if the subject is sufficient
- Wrap at 72 characters

### Examples

```
feat: add periodic publisher with jthread support

fix: resolve ZeroMQ socket timeout on high load

docs: update Docker development instructions

refactor: extract message serialization to separate module

Why: Improve code organization and testability
How: Move serialize/deserialize methods to core/serialization.hpp

test: add unit tests for IMU data validation

chore: update vcpkg baseline to latest

perf: optimize FlatBuffers serialization buffer reuse
```

## Pull Request Commit Guidelines

When submitting pull requests, follow these cost-effective rules to keep the commit history clean and reviewable:

### Squash Small Changes

- Combine related small commits into a single meaningful commit
- Avoid commits like "fix typo", "oops", "WIP" in the final PR
- Use `git rebase -i` to squash before submitting

### One Logical Change Per Commit

- Each commit should represent one complete, working change
- Don't mix unrelated changes in a single commit
- If a commit requires another commit to work, squash them together

### Keep PRs Focused

- One feature or fix per pull request
- Smaller PRs are easier to review and merge faster
- Split large changes into incremental PRs when possible

### Before Submitting

1. Rebase onto the latest main branch
2. Ensure each commit passes tests independently
3. Review your own diff before requesting review
4. Write a clear PR description summarizing all changes

### PR Title Format

Follow the same conventional commit format for PR titles:

```
<type>: <concise description>
```

Example: `feat: add WebSocket transport layer`

## Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run tests (`cmake --workflow --preset debug`)
5. Commit with conventional commit message
6. Push and create a Pull Request

## Code Style

- Follow modern C++20 idioms
- Use `clang-format` for formatting (Google style, 100 column limit)

## Building

```bash
# Debug build with tests
cmake --workflow --preset debug

# Release build
cmake --workflow --preset release
```
