from fastapi import APIRouter

from app.models.response import ApiResponse
from app.services.health_service import health_service


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=ApiResponse,
    summary="Check application health",
)
async def health_check() -> ApiResponse:
    """
    Check the health status of the application.
    """

    return ApiResponse(
        success=True,
        message="Health check completed.",
        data=await health_service.check(),
    )