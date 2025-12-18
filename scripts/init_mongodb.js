/**
 * MongoDB initialization script for ML Agent Platform
 * 
 * This script is optional - MongoDB collections and indexes are
 * automatically created by the application on startup.
 * 
 * Run this script if you want to pre-create the database structure:
 *   mongosh ml_agent scripts/init_mongodb.js
 * 
 * Or connect and run:
 *   mongosh
 *   use ml_agent
 *   load("scripts/init_mongodb.js")
 */

// Switch to the ml_agent database
db = db.getSiblingDB('ml_agent');

print("Initializing ML Agent MongoDB database...");

// ============================================
// Collection: active_jobs
// Purpose: Index table for scheduler to efficiently query running tasks
// ============================================
db.createCollection("active_jobs", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["thread_id", "current_phase", "status"],
            properties: {
                thread_id: {
                    bsonType: "string",
                    description: "Unique identifier for this pipeline run"
                },
                current_phase: {
                    bsonType: "string",
                    enum: ["pending", "sft", "quantization", "evaluation", "deployment", "completed", "error", "cancelled"],
                    description: "Current phase in the pipeline"
                },
                remote_job_id: {
                    bsonType: ["string", "null"],
                    description: "ID of the currently running remote task"
                },
                status: {
                    bsonType: "string",
                    enum: ["pending", "running", "completed", "failed", "cancelled"],
                    description: "Job lifecycle status"
                },
                error_message: {
                    bsonType: ["string", "null"],
                    description: "Error message if status is failed"
                },
                retry_count: {
                    bsonType: "int",
                    minimum: 0,
                    description: "Number of retries attempted"
                },
                last_checked_at: {
                    bsonType: "date",
                    description: "Last time this job was checked by scheduler"
                },
                created_at: {
                    bsonType: "date",
                    description: "When this record was created"
                },
                updated_at: {
                    bsonType: "date",
                    description: "When this record was last updated"
                }
            }
        }
    }
});

print("Created active_jobs collection with schema validation");

// Create indexes for efficient querying
db.active_jobs.createIndex(
    { "thread_id": 1 },
    { unique: true, name: "idx_thread_id" }
);

db.active_jobs.createIndex(
    { "status": 1, "last_checked_at": 1 },
    { name: "idx_status_last_checked" }
);

db.active_jobs.createIndex(
    { "current_phase": 1 },
    { name: "idx_current_phase" }
);

print("Created indexes for active_jobs collection");

// ============================================
// Note: LangGraph Checkpoint Collections
// ============================================
// The following collections are automatically created by LangGraph's MongoDBSaver:
// - checkpoints: Stores serialized graph state
// - checkpoint_writes: Stores pending writes
//
// Do NOT manually create these collections; let LangGraph handle them
// via MongoDBSaver initialization.

print("\nMongoDB initialization complete!");
print("\nNote: LangGraph checkpoint collections will be auto-created on first use.");

