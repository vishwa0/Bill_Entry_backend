from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import json
import os
import time
from litellm import completion
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO

load_dotenv()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://7ddvrn7imiufy5evthycf9o1vh2mccij.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

def configure_gemini():
    """Configure and return Gemini model"""
    return 'gemini/gemini-1.5-flash'

async def process_single_file(file: UploadFile) -> dict:
    """Process a single file with Gemini API"""
    try:
        # Read the prompt
        with open('prompt.txt', 'r') as prompt_file:
            prompt = prompt_file.read()
        
        # Read file content
        file_bytes = await file.read()
        
        # Handle different file types
        if file.content_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            base64_file = base64.b64encode(file_bytes).decode('utf-8')
            image_url = f"data:application/pdf;base64,{base64_file}"
        else:
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        # Make API call to Gemini
        response = completion(
            model=configure_gemini(),
            messages=messages,
            api_key=GOOGLE_API_KEY,
            temperature=0.0,
            top_p=0.95,
            stream=False
        )
        
        # Parse response
        classification_text = response.choices[0].message.content.strip()
        classification_text = classification_text.replace("```json", "").replace("```", "")
        print(classification_text)
        
        try:
            response_json = json.loads(classification_text.split('<answer>')[1].split('</answer>')[0])
            if isinstance(response_json, list) and len(response_json) > 0:
                response_json = response_json[0]
            return response_json
        except Exception as e:
            print(f"Error parsing JSON response: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

@app.post("/api/process-invoices")
async def process_invoices(files: List[UploadFile] = File(...)):
    """Process multiple invoice files"""
    try:
        results = []
        for file in files:
            # Add delay for rate limiting
            time.sleep(1)
            result = await process_single_file(file)
            if result:
                results.append(result)
        
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
