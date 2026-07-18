# Security Policy

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the repository owner.
Do not create public GitHub issues for security problems, exposed secrets, or
deployment-sensitive configuration.

When reporting, include:

- Affected version or commit.
- A clear description of the vulnerability.
- Steps to reproduce, proof of concept, or relevant logs.
- Any known impact or workaround.

The maintainer will review reports and coordinate fixes at a reasonable pace
for a personal open-source project.

## Supported Versions

Only the latest published release and the default branch are supported.

## Secrets

Never commit real `.env` files, API keys, database passwords, object storage
credentials, JWT secrets, or encryption keys. Use `.env.example` files for
safe placeholders only.
