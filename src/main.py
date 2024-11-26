import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
# TODO: 1. ADD REQUIRED IMPORTS (ALSO IN THE REQUIREMENTS.TXT)
import easyocr
import io
import tempfile
import zipfile
import json
import cv2
import numpy as np
from PIL import Image
from table_transformer.src.inference import TableExtractionPipeline

settings = get_settings()


class MyService(Service):
    # TODO: 2. CHANGE THIS DESCRIPTION
    """
    My service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            # TODO: 3. CHANGE THE SERVICE NAME AND SLUG
            name="Table Extraction",
            slug="table-extraction",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            # TODO: 4. CHANGE THE INPUT AND OUTPUT FIELDS, THE TAGS AND THE HAS_AI VARIABLE
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
                FieldDescription(
                    name="layout",
                    type=[
                        FieldDescriptionType.APPLICATION_JSON,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_ZIP]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

    # TODO: 5. CHANGE THE PROCESS METHOD (CORE OF THE SERVICE)
    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.
        image_bytes = data["image"].data
        layout_res = json.loads(data["layout"].data)

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
        image_pil = Image.fromarray(image)

        pipeline = TableExtractionPipeline(str_device="cpu", det_device="cpu",
                                           det_model_path="model/pubtables1m_detection_detr_r18.pth",
                                           str_model_path="model/TATR-v1.1-All-msft.pth")
        # Temporary directory and zip buffer setup
        with tempfile.TemporaryDirectory():
            zip_buffer = io.BytesIO()  # In-memory buffer for ZIP file
            # Create a zipfile in write mode
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, item in enumerate(layout_res):
                    if item["type"] == "table" and item["score"] > 0.5:  # Confidence threshold
                        bbox = item["bbox"]
                        # Crop the image based on the bounding box
                        cropped_image = image_pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        print(cropped_image.size)
                        cropped_image_rgb = cropped_image.convert('RGB')
                        cropped_image_np = np.array(cropped_image_rgb)

                        reader = easyocr.Reader(['en'])

                        # img is a numpy array for your RGB image
                        ocr_result = reader.readtext(cropped_image_np, width_ths=.03)

                        tokens = []
                        for i, res in enumerate(ocr_result):
                            tokens.append({
                                "bbox": list(map(int, [res[0][0][0], res[0][0][1], res[0][2][0], res[0][2][1]])),
                                "text": res[1],
                                "flags": 0,
                                "span_num": i,
                                "line_num": 0,
                                "block_num": 0
                            })

                        # Recognize the table and extract CSV output
                        out_formats = pipeline.recognize(cropped_image, tokens, out_csv=True)
                        print(out_formats)

                        if "csv" in out_formats:
                            # Save CSV content to a file in the zip archive
                            csv_content = out_formats["csv"][0]
                            csv_filename = f"table_{idx}.csv"
                            zf.writestr(csv_filename, csv_content)
                            print(f"Added {csv_filename} to ZIP archive.")

            # Finalize ZIP file
            zip_buffer.seek(0)  # Move the pointer to the start of the buffer

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(data=zip_buffer.read(), type=FieldDescriptionType.APPLICATION_ZIP)
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


# TODO: 6. CHANGE THE API DESCRIPTION AND SUMMARY
api_description = """
Inputs:
- Document Image: An image of the document containing tables (JPEG, PNG).
- Layout Analysis Results: JSON results from a prior layout analysis model, which provides bounding boxes (bboxes)
    for potential tables in the document.

Outputs:
- A ZIP file containing all the detected tables in CSV format.

Model Specifications:
- Model: Table Transformer
- Version: TATR-v1.1-All
- Pretraining Dataset: PubTables-1M
- Finetuning Dataset: FinTabNet
- Model Size: 110 MB
- Reference : [Table Transformer](https://github.com/microsoft/table-transformer)

Capabilities:

    Processes bounding boxes provided by the layout model to crop the regions of interest.
    Extracts table content and structure from cropped images.
    Generates well-structured CSV files from table data.
"""
api_summary = """ This service provides an advanced table extraction solution for image-based documents,
tailored for use cases requiring structured tabular data.
By integrating a pre-trained Table Transformer model,
the service delivers accurate and efficient table detection and extraction.
"""

# Define the FastAPI application with information
# TODO: 7. CHANGE THE API TITLE, VERSION, CONTACT AND LICENSE
app = FastAPI(
    lifespan=lifespan,
    title="Table Extraction",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
