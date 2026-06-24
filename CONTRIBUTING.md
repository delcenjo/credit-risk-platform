# Contributing

This is a personal learning project, but issues and pull requests are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Working on it

- `credit-scoring train` trains and registers a model.
- `credit-scoring serve` runs the API locally.
- `pytest` runs the test suite (offline, no download).
- `ruff check src tests` lints the code.

## Before opening a pull request

- Run `pytest` and `ruff check src tests` and make sure both pass.
- Add or update a test when you change behaviour.
- Keep the style consistent with the surrounding code.
