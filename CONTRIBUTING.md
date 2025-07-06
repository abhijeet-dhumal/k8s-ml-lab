# Contributing

## Quick Start

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes
4. Test: `make setup && make submit-job && make status`
5. Commit and push
6. Create a PR

## Prerequisites

- Docker
- kubectl
- kind
- Python 3.8+

## Testing

```bash
# Full setup and test
make setup
make submit-job
make status
make logs

# Or run complete workflow
make run-e2e-workflow

# Clean up
make cleanup
```

## Key Make Commands

```bash
make setup           # Install deps and create cluster
make run-e2e-workflow # Test complete workflow
make cleanup         # Clean up resources
```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Keep it simple

## Commit Messages

```
feat: add new feature
fix: resolve issue
docs: update documentation
```

That's it! 