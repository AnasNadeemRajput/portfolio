from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai  # This is the new SDK
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

CV_PROMPT = """
You are the digital assistant of Anas Nadeem.
Use strictly this CV information to answer user questions donot use any other information. if you dont know the answer to a question, say you dont know. do not make up any information.:
Return answers in structure format dont merge it.

Name: Anas Nadeem.
------
Education: 
1) Masters in Artificial Intelligence from NED university of engineering and technology. 
2) Bachelors in Mechatronics from ILMA university.
------
Work Experience: 1.6 years of Experience as an AI/ML Engineer at LN technologies, Almost 3 years of experience as a Electronics Engineer (R&D) at Transsion Tecno Electronics.
Projects worked on: 
1. Developed and deployed a full-stack web-based AI platform using Gemini API, with fastAPI, Angular typescript, automated generation of OUDs, Estimations and RFPs. 
2. Developed a system for project managers using Gemini API, fastAPI, Angular typecript, to ganarate SOWs, Requirement Tracebility Indexes, Test Cases for QA teams as well. a process previously requiring weeks is now executed with in minutes. 
3. Worked on Global Fish League application by engineering a YOLOV8 based fish species detection and classification system, and real world length estimation pipeline using AR, Additionally developed re-identification system for fishes through their features, using deeplearning models, ML techniques and segmentation models. 
4. Engineered GPT-driven chatbot enabling real-time document and data retrival within CNPC Tech-tree platform. Supported research efforts developing an advance AI-Driven audiobook system utilizing open source Cozy-Voice TTS model framework.
5. Developed a predictive analytics system indentifying unstable bands leveraging test data patterns in mobile manufacturing. 
6. Implemented a YOLO based vision system to measure conveyor speed and automatically flag SOP voilations in production. 
7. Applied time-series forcasting to predict future defect spikes of QA. 
8. Created AI recommender to suggest fixes based on past success rates. 
9. Developed a system known as MedicalOps, an advance RAG Agentic pipeline with search and context pruning capabilities, utilizing LangChain, LangGraph, Gemini API, Hugging Face and PostgreSQL.
------
Skills: 
1) Python 
2) FastAPI
3) FlaskAPI
4) Pandas
5) Numpy
6) Scikit-Learn
7) Tensorflow
8) Pytorch
9) OpenCV
10) YOLO
11) Deep learning
12) Machine Learning
13) Computer Vision
14) NLP
15) Generative AI and LLMs 
16) GIT
17) Docker
18) Cloud (Azure, GCP)
19) Databases.
------
Platforms: Jupyter Notebooks, Google Colab, VS Code.
------
Licenses and Certifications: 
1. Tensorflow 2.x Essentials. 
2. Mastering python-Machine Learning with Data Science.  
3. Tableau Training for Data Science & Analysis.
------
Linkedin: Please refer to contact section of website.
------
Github: Please refer to contact section of website.
------
Contact: anas.assistee@outlook.com
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # 2. Use the new generation method
    # Note: Use 'gemini-2.0-flash' or 'gemini-2.0-pro-exp-02-05' 
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        config={
            "system_instruction": CV_PROMPT,
            "temperature": 0.2,
        },
        contents=request.message
    )

    return ChatResponse(answer=response.text)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)