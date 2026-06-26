# API App

Backend boundary for auth, workspace, document management, chat, admin, and retrieval traces.

## Planned Stack

- Python
- FastAPI
- Postgres
- Redis queue
- Object storage

## Responsibility

The API should validate requests, enforce tenant isolation, call application services, and return stable response contracts.

Do not put retrieval algorithms directly in route handlers.

