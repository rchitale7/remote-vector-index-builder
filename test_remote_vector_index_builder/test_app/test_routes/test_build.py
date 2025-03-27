# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from core.common.models.index_build_parameters import (
    IndexBuildParameters,
    IndexParameters,
    SpaceType,
    AlgorithmParameters,
)

from app.base.exceptions import HashCollisionError, CapacityError
from app.routes import build

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(build.router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def mock_job_service():
    return Mock()

@pytest.fixture
def index_build_parameters():
    """Create sample IndexBuildParameters for testing"""
    return IndexBuildParameters(
        container_name="testbucket",
        vector_path="test.knnvec",
        doc_id_path="test_ids.txt",
        dimension=3,
        doc_count=2,
        index_parameters=IndexParameters(
            space_type=SpaceType.INNERPRODUCT,
            algorithm_parameters=AlgorithmParameters(
                ef_construction=200, ef_search=200
            ),
        ),
    )

def test_create_job_success(client, mock_job_service, index_build_parameters):
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job_service.create_job.return_value = "test_job_id"

    # Act
    response = client.post("/_build", json=index_build_parameters.model_dump())

    # Assert
    assert response.status_code == 200
    assert response.json() == {"job_id": "test_job_id"}
    mock_job_service.create_job.assert_called_once()

def test_create_job_hash_collision(client, mock_job_service, index_build_parameters):
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job_service.create_job.side_effect = HashCollisionError("Hash collision occurred")

    # Act
    response = client.post("/_build", json=index_build_parameters.model_dump())

    # Assert
    assert response.status_code == 429
    assert response.json() == {"detail": "Hash collision occurred"}

def test_create_job_capacity_error(client, mock_job_service, index_build_parameters):
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job_service.create_job.side_effect = CapacityError("System capacity exceeded")

    # Act
    response = client.post("/_build", json=index_build_parameters.model_dump())

    # Assert
    assert response.status_code == 507
    assert response.json() == {"detail": "System capacity exceeded"}

def test_create_job_unexpected_error(client, mock_job_service, index_build_parameters):
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job_service.create_job.side_effect = Exception("Unexpected error")

    with pytest.raises(Exception):
        client.post("/_build", json=index_build_parameters.model_dump())