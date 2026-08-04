from app.database.database import Base, engine

# Import all models so SQLAlchemy knows about them
from app.database import models


def init_db():
    """
    Create all database tables.
    """
    Base.metadata.create_all(bind=engine)