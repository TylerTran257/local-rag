import logging

from fastapi import APIRouter, Depends, Request

from app.api.schemas import DeleteCollectionRequest, DeleteCollectionResponse
from app.auth import Principal, enforce_scope, require_principal
from app.delete.contracts import DeleteRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents/delete", response_model=DeleteCollectionResponse)
def delete_collection(
    request: Request,
    body: DeleteCollectionRequest,
    principal: Principal = Depends(require_principal),
):
    enforce_scope(
        principal,
        service_name=body.service_name,
        tenant_id=body.tenant_id,
        collections=body.collections,
    )

    result = request.app.state.delete_use_case.execute(
        DeleteRequest(
            service_name=body.service_name,
            tenant_id=body.tenant_id,
            collections=body.collections,
            filters=body.filters,
        )
    )

    return DeleteCollectionResponse(deleted_count=result.deleted_count)
