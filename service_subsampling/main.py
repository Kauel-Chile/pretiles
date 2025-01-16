import os
import shutil
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from utils import subsample_point_cloud, run_potree_converter, zip_folder, upload_to_blob_storage
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), factor: int = 10):
    if file.content_type not in ["application/vnd.las", "application/vnd.laz"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only .las or .laz files are allowed.")
    
    # Save the uploaded file to a temporary location
    temp_input_file = f"tmp/{file.filename}"
    with open(temp_input_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Define the output file path
    temp_output_file_las = f"tmp/reduced_{file.filename}"
    temp_output_file = f"tmp/potree_{file.filename[:-4]}"
    temp_output_zip = f"tmp/potree_{file.filename[:-4]}.zip"

    # Call the subsample_point_cloud function
    subsample_point_cloud(temp_input_file, temp_output_file_las, factor)
    run_potree_converter(temp_output_file_las, temp_output_file)
    zip_folder(temp_output_file, temp_output_zip)
    upload_to_blob_storage(connection_string, container_name, temp_output_zip, f"potree_{file.filename[:-4]}.zip")

    shutil.rmtree("tmp")
    os.makedirs("tmp")
    
    return JSONResponse(content={"filename": f"reduced_{file.filename}", "content_type": file.content_type})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)