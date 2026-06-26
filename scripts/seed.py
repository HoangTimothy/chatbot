import os
import sys

# Add project root, apps/api, and packages/shared to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "api")))

from app.adapters.db.session import SessionLocal
from app.routes.auth import get_password_hash
from shared.enums import UserRole
from shared.models import User, UserWorkspaceRole, Workspace


def seed_database():
    """Seed the database with default workspace and admin user."""
    db = SessionLocal()
    try:
        # Check if default workspace exists
        workspace = db.query(Workspace).filter(Workspace.name == "Default Workspace").first()
        if not workspace:
            workspace = Workspace(name="Default Workspace")
            db.add(workspace)
            db.flush()
            print(f"Created Workspace: '{workspace.name}' (ID: {workspace.id})")
        else:
            print(f"Workspace already exists: '{workspace.name}' (ID: {workspace.id})")

        # Check if default admin user exists
        admin_email = "admin@enterprise-rag.com"
        admin_password = "admin-password-123"
        user = db.query(User).filter(User.email == admin_email).first()
        if not user:
            user = User(
                email=admin_email,
                hashed_password=get_password_hash(admin_password),
                fullname="System Administrator",
            )
            db.add(user)
            db.flush()
            print(f"Created Admin User: '{user.email}' (Password: {admin_password})")
        else:
            print(f"Admin User already exists: '{user.email}'")

        # Associate user with workspace as Owner
        role = (
            db.query(UserWorkspaceRole)
            .filter(
                UserWorkspaceRole.user_id == user.id,
                UserWorkspaceRole.workspace_id == workspace.id,
            )
            .first()
        )
        if not role:
            role = UserWorkspaceRole(
                user_id=user.id,
                workspace_id=workspace.id,
                role=UserRole.OWNER,
            )
            db.add(role)
            print(f"Associated user with workspace as {UserRole.OWNER.value.upper()}")
        else:
            print(f"User is already associated with workspace as role: {role.role.value}")

        db.commit()
        print("\nSeeding completed successfully!")
        print("-" * 50)
        print("Credentials for local testing:")
        print(f"Workspace ID: {workspace.id}")
        print(f"Email:        {user.email}")
        print(f"Password:     {admin_password}")
        print("-" * 50)

    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding database...")
    seed_database()
