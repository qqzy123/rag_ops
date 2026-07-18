# Contributing

This project is currently published primarily for source availability,
learning, evaluation, and personal maintenance by zhoujian.

External contributions are not actively solicited yet. Pull requests may be
reviewed at the maintainer's discretion, but there is no guaranteed response
time or acceptance path.

By submitting a contribution, you confirm that:

- You have the right to submit the code or documentation.
- Your contribution can be licensed under the Apache License, Version 2.0.
- Your submission does not include secrets, private data, or third-party code
  that cannot be redistributed under compatible terms.

Before opening a pull request, run the relevant tests:

```powershell
cd backend
python -m pytest
```

```powershell
cd frontend
npm run typecheck
npm run build
```
