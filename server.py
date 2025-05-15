from fastapi import FastAPI, Request
from pydantic import BaseModel
from re import search, escape
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from fastapi.middleware.cors import CORSMiddleware



# --- Настройка FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модель запроса
class QueryRequest(BaseModel):
    question: str

# --- Список сериалов
series_list = [
    "Stargate SG-1", "Stargate Atlantis", "Star Trek TOS", "Star Trek TNG",
    "Star Trek DS9", "Star Trek Voyager", "Doctor Who", "Babylon 5",
    "Fringe", "Futurama", "Battlestar Galactica", "Firefly"
]

series_keywords = {
    "Stargate SG-1": ["sg-1", "jack o'neill", "teal'c", "samantha carter", "daniel jackson", "stargate command", "goa'uld", "tok'ra"],
    "Stargate Atlantis": ["atlantis", "sheppard", "todd the wraith", "weir", "mckay", "taylor emmagan", "wraith"],
    "Star Trek TOS": ["kirk", "spock", "uss enterprise", "bones", "dr. mccoy", "uhura", "sulu"],
    "Star Trek TNG": ["picard", "riker", "data", "worf", "enterprise-d", "beverly crusher", "deanna troi"],
    "Star Trek DS9": ["sisko", "kira", "odo", "quark", "jadzia dax", "dominion", "deep space 9", "wormhole"],
    "Star Trek Voyager": ["janeway", "chakotay", "seven of nine", "neelix", "uss voyager", "delta quadrant"],
    "Doctor Who": ["doctor who", "tardis", "dalek", "cyberman", "time lord", "gallifrey"],
    "Babylon 5": ["babylon 5", "sheridan", "londo mollari", "delenn", "psi corps", "minbari"],
    "Fringe": ["walter bishop", "olivia dunham", "peter bishop", "observers", "fringe division"],
    "Futurama": ["fry", "leela", "bender", "planet express", "professor farnsworth"],
    "Battlestar Galactica": ["adama", "cylon", "galactica", "caprica", "kobol", "colonial fleet"],
    "Firefly": ["serenity", "malcolm reynolds", "zoe washburne", "jayne cobb", "river tam"]
}

# --- Функция определения сериала
def detect_series(text):
    text = text.lower()
    scores = {s: 0 for s in series_keywords}
    for s, keywords in series_keywords.items():
        for kw in keywords:
            if search(r'\b' + escape(kw) + r'\b', text):
                scores[s] += 1
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Unknown"

# --- Загрузка эмбеддингов
embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open("data/story_texts.pkl", "rb") as f:
    story_nums, texts, series = pickle.load(f)
index = faiss.read_index("data/story_index.faiss")

# --- Функция генерации ответа через Ollama
def generate_answer(context, question):
    prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=600
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        return result or "Модель не вернула ответ."
    except Exception as e:
        return f"Ошибка при обращении к Ollama API: {e}"

# --- Основной endpoint
@app.post("/ask")
def ask_question(query: QueryRequest):
    question = query.question
    detected = detect_series(question)
    print(f"Сериал определён: {detected}")

    query_emb = embedder.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_emb).astype("float32"), 20)

    # Фильтрация по сериалу
    filtered = [i for i in indices[0] if series[i] == detected or detected == "Unknown"]
    top_k = filtered[:3]

    # Обрезаем каждую историю до 1500 символов
    context = "\n\n".join(texts[i][:1500] for i in top_k)

    print(f"️ Используемые фрагменты: {[series[i] for i in top_k]}")
    answer = generate_answer(context, question)

    return {
        "series": detected,
        "context_sources": [series[i] for i in top_k],
        "answer": answer
    }
