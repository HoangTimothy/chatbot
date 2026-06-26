from datetime import datetime
from typing import Annotated, List

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.db.session import get_db_session
from app.routes.auth import get_current_user, get_password_hash
from shared.enums import UserRole
from shared.models import User, UserWorkspaceRole, Workspace

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceResponse(BaseModel):
    """Pydantic model representing Workspace details."""

    id: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


class WorkspaceUserResponse(BaseModel):
    """Pydantic model representing workspace member user information and role."""

    user_id: str
    email: str
    fullname: str | None
    role: UserRole

    class Config:
        from_attributes = True


class InviteUserRequest(BaseModel):
    """Pydantic model representing user invitation fields."""

    email: str
    role: UserRole
    fullname: str | None = None


async def get_current_workspace(
    x_workspace_id: Annotated[str, Header(description="Workspace context ID")],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> Workspace:
    """Dependency that resolves workspace ID from header and verifies user access."""
    # Find workspace
    stmt = select(Workspace).where(Workspace.id == x_workspace_id)
    result = await db.execute(stmt)
    workspace = result.scalars().first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    # Validate membership role exists
    role_stmt = select(UserWorkspaceRole).where(
        UserWorkspaceRole.user_id == current_user.id,
        UserWorkspaceRole.workspace_id == workspace.id,
    )
    role_result = await db.execute(role_stmt)
    user_role = role_result.scalars().first()
    if not user_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to workspace denied",
        )

    # Attach current user's role context to workspace object for easy route level auth checks
    workspace.current_user_role = user_role.role
    return workspace


@router.get("/current", response_model=WorkspaceResponse)
async def get_current(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
):
    """Fetch details of the current active workspace."""
    return workspace


@router.get("/current/users", response_model=List[WorkspaceUserResponse])
async def get_workspace_users(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    """Fetch all users having roles and access to the current workspace."""
    stmt = (
        select(User.id, User.email, User.fullname, UserWorkspaceRole.role)
        .join(UserWorkspaceRole, User.id == UserWorkspaceRole.user_id)
        .where(UserWorkspaceRole.workspace_id == workspace.id)
    )
    result = await db.execute(stmt)
    users = []
    for row in result.all():
        users.append(
            WorkspaceUserResponse(
                user_id=row[0], email=row[1], fullname=row[2], role=row[3]
            )
        )
    return users


@router.post("/current/users/invite", status_code=status.HTTP_201_CREATED)
async def invite_user(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    request: InviteUserRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    """Invite a user to the current workspace (Admin and Owner only)."""
    # Enforce role logic
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace Owners and Admins can invite new users",
        )

    # Check if user already exists
    user_stmt = select(User).where(User.email == request.email)
    user_result = await db.execute(user_stmt)
    user = user_result.scalars().first()

    if not user:
        # Create a new placeholder user
        # In a real system, an invitation email would be sent to complete registration
        # Here we create a placeholder user with a random temporary password
        temp_pwd = get_password_hash("temp-password-123")
        user = User(
            email=request.email,
            hashed_password=temp_pwd,
            fullname=request.fullname,
        )
        db.add(user)
        await db.flush()  # to populate user.id

    # Check if membership already exists
    existing_role_stmt = select(UserWorkspaceRole).where(
        UserWorkspaceRole.user_id == user.id,
        UserWorkspaceRole.workspace_id == workspace.id,
    )
    existing_role_result = await db.execute(existing_role_stmt)
    existing_role = existing_role_result.scalars().first()

    if existing_role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists in this workspace",
        )

    # Add role to workspace
    new_role = UserWorkspaceRole(
        user_id=user.id,
        workspace_id=workspace.id,
        role=request.role,
    )
    db.add(new_role)
    await db.commit()

    return {"detail": f"User {request.email} successfully added as {request.role}"}
