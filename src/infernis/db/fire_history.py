"""Historical fire database table and query helpers."""

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import Column, Date, DateTime, Float, Index, Integer, String

from infernis.db.engine import Base


class FireHistoryDB(Base):
    """Historical fire occurrence records from CNFDB / BC Fire Perimeters."""

    __tablename__ = "fire_history"

    id = Column(Integer, primary_key=True)
    fire_id = Column(String(50), unique=True, nullable=False)
    fire_name = Column(String(200))
    year = Column(Integer, nullable=False, index=True)
    start_date = Column(Date)
    end_date = Column(Date)
    cause = Column(String(50))  # human, lightning, unknown
    size_ha = Column(Float)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    geom = Column(Geometry("POINT", srid=4326))
    source = Column(String(30))  # cnfdb, bc_fire_perimeters, bc_fire_incidents

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_fire_history_geom", geom, postgresql_using="gist"),
        Index("ix_fire_history_year", year),
    )
