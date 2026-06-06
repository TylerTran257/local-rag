from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile

from app.services.document_service import JobData

router = APIRouter()


@router.post("/upload_async", status_code=202)
async def upload_async(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> JobData:
    document = await request.app.state.document_service.create_document(file)
    job = request.app.state.document_service.create_job(document["document_id"])
    background_tasks.add_task(
        request.app.state.document_service.run_indexing_pipeline,
        document["document_id"],
        job["job_id"],
    )

    return job


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> JobData:
    job = request.app.state.document_service.get_job(job_id)

    return job
