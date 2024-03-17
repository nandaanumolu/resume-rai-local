from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import get_summary_exp_skills,get_jobdescription,get_resumes

app = FastAPI()

class RequestBody(BaseModel):
    context: str
    category: str
    threshold: float
    noOfMatches: int
    inputPath: str

@app.post("/process/")
async def process_request(body: RequestBody):
    # Check if category is valid
    if body.category not in ['resume', 'job']:
        raise HTTPException(status_code=400, detail="Category must be 'resume' or 'job'")
    if body.category == 'resume':
        get_summary_exp_skills(body.inputPath)
        description = get_jobdescription(body.context)
        resumes = get_resumes(description,body.threshold,body.noOfMatches)
        resumes['status'] = 'success'
# Process the request here
# For demonstration, let's just return the request body
        return resumes


