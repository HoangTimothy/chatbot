import os
import sys

# Add project paths to sys.path to allow direct python execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "api")))

import pytest
from fastapi.testclient import TestClient

from app.adapters.db.session import SessionLocal
from app.main import app
from shared.models import User, Workspace

client = TestClient(app)


def test_health():
    """Verify backend health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_auth_and_workspace():
    """Verify JWT authentication, token profile fetching, and workspace access checks."""
    db = SessionLocal()
    try:
        # Resolve the seeded admin credentials and workspace ID
        user = db.query(User).filter(User.email == "admin@enterprise-rag.com").first()
        workspace = db.query(Workspace).filter(Workspace.name == "Default Workspace").first()

        assert user is not None
        assert workspace is not None

        # 1. Login verification
        login_data = {
            "username": "admin@enterprise-rag.com",
            "password": "admin-password-123",
        }
        response = client.post("/auth/login", data=login_data)
        assert response.status_code == 200
        token_data = response.json()
        assert "access_token" in token_data
        token = token_data["access_token"]

        # 2. Get current user verification
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        user_data = response.json()
        assert user_data["email"] == "admin@enterprise-rag.com"

        # 3. Workspace detail fetch verification
        headers_with_ws = {
            "Authorization": f"Bearer {token}",
            "X-Workspace-ID": workspace.id,
        }
        response = client.get("/workspaces/current", headers=headers_with_ws)
        assert response.status_code == 200
        ws_data = response.json()
        assert ws_data["name"] == "Default Workspace"
        assert ws_data["id"] == workspace.id

        # 4. Workspace membership fetch verification
        response = client.get("/workspaces/current/users", headers=headers_with_ws)
        assert response.status_code == 200
        users_list = response.json()
        assert len(users_list) >= 1
        assert any(u["email"] == "admin@enterprise-rag.com" for u in users_list)

    finally:
        db.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))

