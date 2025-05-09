# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
from unittest.mock import Mock, patch

from app.base.exceptions import HashCollisionError, CapacityError
from app.models.job import Job
from app.models.workflow import BuildWorkflow
from app.services.job_service import JobService


@pytest.fixture
def request_store():
    mock = Mock()
    mock.get.return_value = None
    mock.add.return_value = True
    mock.delete.return_value = True
    return mock


@pytest.fixture
def total_gpu_memory():
    return 1000.0


@pytest.fixture
def total_cpu_memory():
    return 1000.0


@pytest.fixture
def workflow_executor():
    mock = Mock()
    mock.submit_workflow.return_value = None
    return mock


@pytest.fixture
def job_service(request_store, workflow_executor, total_gpu_memory, total_cpu_memory):
    return JobService(
        request_store=request_store,
        workflow_executor=workflow_executor,
        total_gpu_memory=total_gpu_memory,
        total_cpu_memory=total_cpu_memory,
    )


@pytest.fixture
def mock_request_parameters():
    return Mock()


@pytest.fixture
def mock_job():
    job = Mock(spec=Job)
    job.compare_request_parameters.return_value = True
    return job


def test_validate_job_existence_no_job(job_service, mock_request_parameters):
    """Test job validation when job doesn't exist"""
    assert (
        job_service._validate_job_existence("test_id", mock_request_parameters) is False
    )


def test_validate_job_existence_matching_job(
    job_service, mock_request_parameters, mock_job
):
    """Test job validation with existing matching job"""
    job_service.request_store.get.return_value = mock_job
    assert (
        job_service._validate_job_existence("test_id", mock_request_parameters) is True
    )


def test_validate_job_existence_hash_collision(
    job_service, mock_request_parameters, mock_job
):
    """Test job validation with hash collision"""
    mock_job.compare_request_parameters.return_value = False
    job_service.request_store.get.return_value = mock_job

    with pytest.raises(HashCollisionError):
        job_service._validate_job_existence("test_id", mock_request_parameters)


@patch("app.services.job_service.Job")
def test_add_to_request_store_success(job, job_service, mock_request_parameters):
    """Test successful addition to request store"""
    job_service._add_to_request_store("test_id", mock_request_parameters)
    job_service.request_store.add.assert_called_once()


@patch("app.services.job_service.Job")
def test_add_to_request_store_failure(job, job_service, mock_request_parameters):
    """Test failed addition to request store"""
    job_service.request_store.add.return_value = False

    with pytest.raises(CapacityError):
        job_service._add_to_request_store("test_id", mock_request_parameters)


def test_create_workflow_success(
    job_service, index_build_parameters, total_gpu_memory, total_cpu_memory
):
    """Test successful workflow creation"""
    workflow = job_service._create_workflow(
        "test_id",
        total_gpu_memory - 100,
        total_cpu_memory - 100,
        index_build_parameters,
    )
    assert isinstance(workflow, BuildWorkflow)
    assert workflow.job_id == "test_id"


def test_create_workflow_gpu_memory_capacity_failure(
    job_service, index_build_parameters, total_gpu_memory, total_cpu_memory
):
    """Test workflow creation with gpu memory capacity failure"""

    with pytest.raises(CapacityError):
        job_service._create_workflow(
            "test_id",
            total_gpu_memory + 100,
            total_cpu_memory - 100,
            index_build_parameters,
        )
        job_service.request_store.delete.assert_called_once()


def test_create_workflow_cpu_memory_capacity_failure(
    job_service, index_build_parameters, total_gpu_memory, total_cpu_memory
):
    """Test workflow creation with cpu memory capacity failure"""

    with pytest.raises(CapacityError):
        job_service._create_workflow(
            "test_id",
            total_gpu_memory - 100,
            total_cpu_memory + 100,
            index_build_parameters,
        )
        job_service.request_store.delete.assert_called_once()


@patch("app.services.job_service.create_request_parameters")
@patch("app.services.job_service.generate_job_id")
@patch("app.services.job_service.calculate_memory_requirements")
@patch("app.services.job_service.Job")
def test_create_job_success(
    job,
    mock_calc,
    mock_generate_id,
    mock_create_params,
    job_service,
    index_build_parameters,
):
    """Test successful job creation"""
    mock_calc.return_value = (100.0, 200.0)
    mock_generate_id.return_value = "test_id"
    mock_create_params.return_value = Mock()

    job_id = job_service.create_job(index_build_parameters)

    assert job_id == "test_id"
    job_service.workflow_executor.submit_workflow.assert_called_once()


@patch("app.services.job_service.create_request_parameters")
@patch("app.services.job_service.generate_job_id")
def test_create_job_exists(
    mock_generate_id, mock_create_params, job_service, index_build_parameters
):
    """Test successful job creation"""
    mock_generate_id.return_value = "test_id"
    mock_create_params.return_value = Mock()

    job_service.request_store.get.return_value = Mock(spec=Job)
    job_id = job_service.create_job(index_build_parameters)

    assert job_id == "test_id"
    job_service.workflow_executor.submit_workflow.assert_not_called()


def test_get_job_exists(job_service, mock_job):
    """Test retrieving existing job"""
    job_service.request_store.get.return_value = mock_job
    result = job_service.get_job("test_id")
    assert result == mock_job


def test_get_job_not_exists(job_service):
    """Test retrieving non-existent job"""
    result = job_service.get_job("test_id")
    assert result is None


def test_get_jobs(job_service):
    """Test get all jobs"""

    mock_job_1 = Mock(spec=Job)
    mock_job_2 = Mock(spec=Job)

    job_service.request_store.get_jobs.return_value = {
        "test_id1": mock_job_1,
        "test_id2": mock_job_2,
    }

    jobs = job_service.get_jobs()

    assert "test_id1" in jobs
    assert "test_id2" in jobs
    assert len(jobs) == 2
    assert jobs["test_id1"] == mock_job_1
    assert jobs["test_id2"] == mock_job_2
