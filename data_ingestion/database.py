"""
Database connection and session management
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator

from .models import Base


# Database URL from environment
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://user:password@localhost:5432/football_betting_ai'
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=os.getenv('DEBUG', 'false').lower() == 'true'
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")


def drop_db():
    """Drop all tables - USE WITH CAUTION"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️  All database tables dropped")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Database session context manager
    
    Usage:
        with get_db() as db:
            db.query(Team).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get database session for FastAPI dependency injection
    
    Usage in FastAPI:
        @app.get("/")
        def endpoint(db: Session = Depends(get_db_session)):
            return db.query(Team).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
