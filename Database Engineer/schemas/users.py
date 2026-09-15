from datetime import datetime

from pydantic import BaseModel, EmailStr, ConfigDict

from database.models import UserRole


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    role: UserRole = UserRole.viewer


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    role: UserRole
    created_at: datetime
