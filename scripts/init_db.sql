-- Database initialization script for ML Agent Platform (PostgreSQL backend)
--
-- NOTE: This script is only needed if using PostgreSQL storage backend.
-- The default storage backend is MongoDB, which auto-creates collections.
--
-- To use PostgreSQL, set STORAGE_BACKEND=postgres in your .env file.
-- Then run this script to set up the database schema.

-- ============================================
-- Extension setup
-- ============================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- Table: active_jobs
-- Purpose: Index table for scheduler to efficiently query running tasks
-- This is separate from LangGraph's checkpoint table for performance
-- ============================================
CREATE TABLE IF NOT EXISTS active_jobs (
    -- Primary identifier, matches LangGraph's thread_id
    thread_id VARCHAR(255) PRIMARY KEY,
    
    -- Current pipeline phase for quick filtering
    current_phase VARCHAR(50) NOT NULL DEFAULT 'pending',
    
    -- Remote job tracking (when waiting for external task completion)
    remote_job_id VARCHAR(255),
    
    -- Job lifecycle status
    -- 'pending': awaiting first execution or next phase
    -- 'running': actively being processed or waiting for remote task
    -- 'completed': all phases finished successfully
    -- 'failed': encountered unrecoverable error
    -- 'cancelled': manually stopped by user
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    
    -- Timestamps for monitoring and debugging
    last_checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata for debugging and auditing
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

-- Index for scheduler polling - most common query pattern
CREATE INDEX IF NOT EXISTS idx_active_jobs_status 
    ON active_jobs(status) 
    WHERE status IN ('running', 'pending');

-- Index for phase-based filtering
CREATE INDEX IF NOT EXISTS idx_active_jobs_phase 
    ON active_jobs(current_phase);

-- ============================================
-- Function: Update timestamp trigger
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to active_jobs
DROP TRIGGER IF EXISTS update_active_jobs_updated_at ON active_jobs;
CREATE TRIGGER update_active_jobs_updated_at
    BEFORE UPDATE ON active_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Note: LangGraph Checkpoint Tables
-- ============================================
-- The following tables are automatically created by LangGraph's PostgresSaver:
-- - checkpoints: Stores serialized graph state
-- - checkpoint_writes: Stores pending writes
-- - checkpoint_blobs: Stores large binary data
--
-- Do NOT manually create these tables; let LangGraph handle them
-- via PostgresSaver.setup() method.

