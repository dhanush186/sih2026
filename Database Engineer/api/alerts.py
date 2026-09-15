from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import Alert, AlertStatus, AlertSeverity
from schemas.alerts import AlertOut, AlertStatusUpdate

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("/", response_model=list[AlertOut])
def list_alerts(
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[AlertStatus] = None,
    severity: Optional[AlertSeverity] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Alert)
    if status_filter is not None:
        query = query.filter(Alert.status == status_filter)
    if severity is not None:
        query = query.filter(Alert.severity == severity)
    return query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found.")
    return alert


@router.patch("/{alert_id}", response_model=AlertOut)
def update_alert_status(alert_id: int, payload: AlertStatusUpdate, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found.")
    alert.status = payload.status
    db.commit()
    db.refresh(alert)
    return alert
