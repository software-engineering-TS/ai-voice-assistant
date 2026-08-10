from unittest.mock import AsyncMock

import pytest

import app.api.routes.health as health_module


@pytest.mark.asyncio
async def test_health_check():

    expected = {
        "status": "healthy",
        "database": "healthy",
    }

    health_module.health_service.check = AsyncMock(
        return_value=expected
    )

    result = await health_module.health_check()

    assert result.success is True

    assert result.message == (
        "Health check completed."
    )

    assert result.data == expected

    health_module.health_service.check.assert_awaited_once_with()