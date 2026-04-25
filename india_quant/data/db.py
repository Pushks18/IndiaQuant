"""Database connection, session factory, and schema creation."""
import argparse
from contextlib import contextmanager

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from india_quant.config import cfg
from india_quant.data.models import Base


_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            cfg.database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_session() -> Session:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all():
    """Create schema. Enables TimescaleDB hypertables only if the extension is present
    — vanilla Postgres works fine for everything we use."""
    engine = get_engine()
    logger.info("Creating all tables...")
    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        timescale_ok = False
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            conn.commit()
            timescale_ok = True
            logger.info("TimescaleDB extension enabled.")
        except Exception:
            conn.rollback()
            logger.info("TimescaleDB unavailable — running on vanilla Postgres (fine).")

        if timescale_ok:
            try:
                conn.execute(text(
                    "SELECT create_hypertable('price_data', 'datetime', "
                    "if_not_exists => TRUE, migrate_data => TRUE);"
                ))
                conn.commit()
                logger.info("Hypertable created for price_data.")
            except Exception as e:
                conn.rollback()
                logger.info(f"Hypertable creation skipped: {e}")

    logger.info("Database schema ready.")


def drop_all():
    engine = get_engine()
    logger.warning("Dropping all tables!")
    Base.metadata.drop_all(engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create", "drop"])
    args = parser.parse_args()

    if args.command == "create":
        create_all()
    elif args.command == "drop":
        drop_all()
