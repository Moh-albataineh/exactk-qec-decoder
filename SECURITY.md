# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

- **Email**: [your-email@example.com]
- **Do not** open a public GitHub issue for security vulnerabilities

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Scope

This is a research software package for quantum error correction simulation. It does not handle user authentication, personal data, or network services. Security concerns are primarily:

- Dependency vulnerabilities (PyTorch, NumPy, Stim)
- Path traversal in file I/O operations
- Unintended data exposure in generated artifacts
