# Contributing to VideoRAG

Thank you for your interest in contributing to VideoRAG!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/AyhamJo7/VideoRAG.git
cd VideoRAG
```

2. Set up development environment:
```bash
make setup
source venv/bin/activate
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Standards

- Python 3.11+
- Follow PEP 8 style guide
- Use type hints wherever possible
- Write docstrings for public APIs (Google style)
- Keep line length ≤ 100 characters

## Testing

- Write tests for new features
- Ensure all tests pass: `make test`
- Aim for >80% code coverage

## Pull Request Process

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes and commit using Conventional Commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `refactor:` code refactoring
   - `test:` adding tests
   - `chore:` maintenance tasks
3. Run formatters and linters: `make format && make lint`
4. Push and create a PR with a clear description
5. Ensure CI passes

## Reporting Issues

- Use GitHub Issues
- Provide reproducible steps
- Include system info and logs
