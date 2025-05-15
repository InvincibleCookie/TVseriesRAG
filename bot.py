from re import search, escape

def detect_series(text):
    text = text.lower()
    scores = {s: 0 for s in series_list}
    keywords = {
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
    for s in scores:
        for kw in keywords[s]:
            if search(r'\b' + escape(kw) + r'\b', text):
                scores[s] += 1
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Unknown"
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from model import generate_answer
from config import TELEGRAM_TOKEN

series_list = [
    "Stargate SG-1", "Stargate Atlantis", "Star Trek TOS", "Star Trek TNG",
    "Star Trek DS9", "Star Trek Voyager", "Doctor Who", "Babylon 5",
    "Fringe", "Futurama", "Battlestar Galactica", "Firefly"
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open("data/story_texts.pkl", "rb") as f:
    story_nums, texts, series = pickle.load(f)
index = faiss.read_index("data/story_index.faiss")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    detected_series = detect_series(query)
    print(f"Определён сериал: {detected_series}")

    q_emb = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(q_emb).astype("float32"), 20)

    # Фильтрация по сериалу
    filtered = [i for i in indices[0] if series[i] == detected_series or detected_series == "Unknown"]
    top_k = filtered[:3]

    context = "".join([texts[i] for i in top_k])
    print(context)
    for i in top_k:
        print(f"В контексте будет использован фрагмент из: {series[i]}")

    response = generate_answer(context, query)
    await update.message.reply_text(response)

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    intro = "Это бот по научно-фантастическим сериалам. Задай вопрос, и я постараюсь найти ответ по сюжетам из популярных шоу."
    known_series = "\n".join(f"- {s}" for s in series_list)
    await update.message.reply_text(f"{intro}\n\nСериалы, которые я знаю:\n{known_series}")

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", handle_start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.run_polling()
