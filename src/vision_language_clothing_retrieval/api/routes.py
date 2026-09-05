import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from vision_language_clothing_retrieval.api.schemas import (
    QueryType,
    SearchResponse, SearchResultItem
)

from vision_language_clothing_retrieval.services.retrieval_service import (
    RetrievalService,
)

router = APIRouter()

def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service

@router.get("/health")
def health() -> dict:
    return {"status": "ok"}

@router.post("/search", response_model=SearchResponse)
async def search(
    query_type: QueryType = Form(...),
    text: str | None = Form(None),
    image: UploadFile | None = File(None),
    top_k: int = Form(5),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """
        - query_type="text"  -> polje 'text' je obavezno, vraćaju se slike.
        - query_type="image" -> fajl 'image' je obavezan, vraćaju se opisi.
    """

    if query_type == QueryType.TEXT:
        if not text:
            raise HTTPException(
                status_code=422,
                detail="Polje 'text' je obavezno kada je query_type='text'."
            )

        try:
            raw_results = retrieval_service.retrieve_images(text, top_k=top_k)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        results = [
            {
                "sample_id": r["sample_id"],
                "score": r["score"],
                "result_type": "image",
                "content": f"/images/{Path(r['image_path']).name}",
            }
            for r in raw_results
        ]

    else:
        if image is None:
            raise HTTPException(
                status_code=422,
                detail="Fajl 'image' je obavezan kada je query_type='image'.",
            )

        filename = image.filename or "upload.jpg"
        temp_path = Path(tempfile.gettempdir()) / filename
        temp_path.write_bytes(await image.read())

        try:
            raw_results = retrieval_service.retrieve_texts(
                str(temp_path), top_k=top_k
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        results = [
            {
                "sample_id": r["sample_id"],
                "score": r["score"],
                "result_type": "text",
                "content": r["text"],
            }
            for r in raw_results
        ]

    return SearchResponse(query_type=query_type, results=results)