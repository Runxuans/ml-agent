"""
Database setup script.

Creates the database and runs initialization SQL.
Run this before starting the application.

Usage:
    python -m scripts.setup_db
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from core.config import settings


async def setup_database():
    """Create database and tables."""
    
    # Parse connection info
    db_url = settings.database_url
    # Extract database name from URL
    # Format: postgresql+asyncpg://user:pass@host:port/dbname
    db_name = db_url.split("/")[-1]
    base_url = "/".join(db_url.split("/")[:-1])
    
    # Connect to postgres database to create our database
    postgres_url = f"{base_url}/postgres".replace("+asyncpg", "")
    
    print(f"Connecting to PostgreSQL...")
    print(f"Database URL: {db_url}")
    
    try:
        # Connect to default database
        conn = await asyncpg.connect(postgres_url.replace("postgresql://", "postgres://"))
        
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            db_name
        )
        
        if not exists:
            print(f"Creating database: {db_name}")
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database created: {db_name}")
        else:
            print(f"Database already exists: {db_name}")
        
        await conn.close()
        
        # Connect to our database and run init script
        app_conn = await asyncpg.connect(
            db_url.replace("postgresql+asyncpg://", "postgres://")
        )
        
        # Read and execute init script
        init_script = project_root / "scripts" / "init_db.sql"
        if init_script.exists():
            print(f"Running initialization script...")
            sql = init_script.read_text()
            await app_conn.execute(sql)
            print("Initialization script completed")
        else:
            print(f"Warning: {init_script} not found")
        
        await app_conn.close()
        
        print("Database setup complete!")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(setup_database())

