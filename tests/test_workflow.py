"""
Integration tests for the training workflow.

Tests the complete workflow without database persistence.
"""

import asyncio
import pytest

from graphs.state.agent_state import AgentState, JobPhase, JobStatus
from graphs.workflows.training_flow import create_training_workflow
from tools.cluster_api.mock_client import MockClusterClient


@pytest.fixture
def mock_client():
    """Create a mock client with very short task duration."""
    return MockClusterClient(task_duration=1)


@pytest.fixture
def workflow(mock_client):
    """Create workflow without checkpointer for testing."""
    return create_training_workflow(
        cluster_client=mock_client,
        checkpointer=None,  # No persistence for unit tests
    )


@pytest.mark.asyncio
async def test_workflow_initial_invoke(workflow):
    """Test that workflow starts correctly."""
    initial_state = {
        "thread_id": "test-001",
        "base_model": "llama3-8b",
        "dataset_path": "s3://test/data.json",
        "config": {"quantization": "int4"},
    }
    
    result = await workflow.invoke(initial_state, "test-001")
    
    # Should have submitted SFT task
    assert result["current_phase"] == JobPhase.SFT.value
    assert result["remote_job_id"] is not None
    # Status can be running or pending depending on when check happens
    assert result["remote_job_status"] in (JobStatus.RUNNING.value, JobStatus.PENDING.value)


@pytest.mark.asyncio
async def test_workflow_full_pipeline(mock_client):
    """Test complete pipeline execution (with very fast mock)."""
    # Use 0.5 second tasks for fast testing
    await mock_client.set_task_duration(1)
    
    workflow = create_training_workflow(
        cluster_client=mock_client,
        checkpointer=None,
    )
    
    initial_state = {
        "thread_id": "test-full-001",
        "base_model": "llama3-8b",
        "dataset_path": "s3://test/data.json",
        "config": {},
    }
    
    # Start workflow
    result = await workflow.invoke(initial_state, "test-full-001")
    
    # Keep invoking until complete
    max_iterations = 50
    iteration = 0
    
    while result.get("current_phase") not in (
        JobPhase.COMPLETED.value,
        JobPhase.ERROR.value,
    ):
        iteration += 1
        if iteration > max_iterations:
            pytest.fail(f"Workflow did not complete after {max_iterations} iterations")
        
        # Wait a bit for mock job to progress
        await asyncio.sleep(0.5)
        
        # Clear needs_wait flag before resuming (simulates scheduler behavior)
        result["needs_wait"] = False
        
        # Resume workflow
        result = await workflow.invoke(result, "test-full-001")
    
    # Verify completion
    assert result["current_phase"] == JobPhase.COMPLETED.value
    assert result["sft_model_path"] is not None
    assert result["quant_model_path"] is not None
    assert result["eval_report"] is not None
    assert result["deploy_url"] is not None


@pytest.mark.asyncio
async def test_workflow_handles_failure(mock_client):
    """Test that workflow handles task failures correctly."""
    workflow = create_training_workflow(
        cluster_client=mock_client,
        checkpointer=None,
    )
    
    initial_state = {
        "thread_id": "test-fail-001",
        "base_model": "llama3-8b",
        "dataset_path": "s3://test/data.json",
        "config": {},
        "max_retries": 0,  # No retries
    }
    
    # Start workflow
    result = await workflow.invoke(initial_state, "test-fail-001")
    job_id = result["remote_job_id"]
    
    # Force the job to fail
    await mock_client.force_job_failure(job_id, "Simulated failure")
    
    # Clear needs_wait flag before resuming (simulates scheduler behavior)
    result["needs_wait"] = False
    
    # Resume workflow
    result = await workflow.invoke(result, "test-fail-001")
    
    # Should be in error state
    assert result["current_phase"] == JobPhase.ERROR.value
    assert result["error_message"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

