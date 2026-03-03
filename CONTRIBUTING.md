# Contributing to ExactK QEC Decoder

Thank you for your interest in contributing!

## How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature or fix: `git checkout -b feature/your-feature`
3. **Write tests** for any new functionality
4. **Run the test suite**: `python -m pytest tests/ -v`
5. **Submit a pull request** with a clear description

## Code Style

- Follow PEP 8 for Python code
- Use type hints where practical
- Add docstrings to public functions

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps and expected vs actual behavior
- Include Python version and OS information

## Development Setup

```bash
git clone <repo-url>
cd exactk-qec-decoder
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Scope

This repository covers the ExactK iso-K hinge loss mechanism and its production deployment pipeline (Selector v6, MLOps hardening). Contributions that maintain backward compatibility and pass the existing test suite are welcome.
