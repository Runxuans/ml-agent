"""
Tests for MockClusterClient.

Verifies that the mock client behaves correctly for testing purposes.
"""

import asyncio
import pytest
from datetime import datetime

from tools.cluster_api.mock_client import MockClusterClient
from tools.cluster_api.base import TaskType, RemoteJobStatus


@pytest.fixture
def mock_client():
    """Create a mock client with short task duration for testing."""
    return MockClusterClient(task_duration=2)  # 2 second tasks


@pytest.mark.asyncio
async def test_submit_task(mock_client):
    """Test submitting a task."""
    job_info = await mock_client.submit_task(
        TaskType.SFT,
        {"base_model": "test-model", "dataset": "test-data"},
    )
    
    assert job_info.job_id.startswith("mock-sft-")
    assert job_info.status == RemoteJobStatus.PENDING
    assert job_info.task_type == TaskType.SFT


@pytest.mark.asyncio
async def test_job_lifecycle(mock_client):
    """Test full job lifecycle: submit -> running -> complete."""
    # Submit
    job_info = await mock_client.submit_task(
        TaskType.SFT,
        {"base_model": "test-model"},
    )
    job_id = job_info.job_id
    
    # Wait for transition to running
    await asyncio.sleep(3)  # Wait past pending delay
    
    status = await mock_client.get_job_status(job_id)
    assert status.status in (RemoteJobStatus.RUNNING, RemoteJobStatus.SUCCESS)
    
    # Wait for completion
    await asyncio.sleep(3)
    
    status = await mock_client.get_job_status(job_id)
    assert status.status == RemoteJobStatus.SUCCESS
    assert status.progress == 100.0
    
    # Get result
    result = await mock_client.get_job_result(job_id)
    assert result.success
    assert result.output_path is not None


@pytest.mark.asyncio
async def test_cancel_job(mock_client):
    """Test cancelling a running job."""
    job_info = await mock_client.submit_task(
        TaskType.QUANTIZATION,
        {"model_path": "test-path"},
    )
    
    # Cancel immediately
    cancelled = await mock_client.cancel_job(job_info.job_id)
    assert cancelled
    
    # Check status
    status = await mock_client.get_job_status(job_info.job_id)
    assert status.status == RemoteJobStatus.CANCELLED


@pytest.mark.asyncio
async def test_force_failure(mock_client):
    """Test forcing a job to fail."""
    job_info = await mock_client.submit_task(
        TaskType.SFT,
        {"base_model": "test"},
    )
    
    # Force failure
    await mock_client.force_job_failure(job_info.job_id, "Test error")
    
    # Check status
    status = await mock_client.get_job_status(job_info.job_id)
    assert status.status == RemoteJobStatus.FAILED
    
    # Get result
    result = await mock_client.get_job_result(job_info.job_id)
    assert not result.success
    assert result.error_message == "Test error"


@pytest.mark.asyncio
async def test_progress_tracking(mock_client):
    """Test that progress updates during execution."""
    # Use longer duration for progress tracking
    await mock_client.set_task_duration(4)
    
    job_info = await mock_client.submit_task(
        TaskType.SFT,
        {"base_model": "test"},
    )
    
    # Wait for running state
    await asyncio.sleep(3)
    
    status = await mock_client.get_job_status(job_info.job_id)
    if status.status == RemoteJobStatus.RUNNING:
        assert status.progress is not None
        assert 0 < status.progress < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

