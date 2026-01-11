from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # استيراد المكتبة المهمة
from engine import RecommenderEngine

app = FastAPI(title="Training Recommender API", version="1.0")

# --- إضافة الـ CORS (هذا اللي يخلي Lovable يشتغل) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يسمح لجميع المواقع بالاتصال بـ API حقك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------

engine = RecommenderEngine()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend/trainee/{trainee_id}")
def recommend_trainee(trainee_id: str, top_k: int = 10):
    return engine.recommend_for_trainee(trainee_id, top_k)

@app.get("/recommend/course/{course_id}")
def recommend_course(course_id: str, top_k: int = 10):
    return engine.recommend_for_course(course_id, top_k)