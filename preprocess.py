from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import pandas as pd
import numpy as np
import faiss
import re
import pickle
import os

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

series_list = list(series_keywords.keys())

def detect_series(text):
    text = text.lower()
    scores = {s: 0 for s in series_keywords}
    for s, keywords in series_keywords.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                scores[s] += 1
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Unknown"

# Создаём папку data, если не существует
os.makedirs("data", exist_ok=True)

dataset = load_dataset("lara-martin/Scifi_TV_Shows", trust_remote_code=True)['train']
stories = defaultdict(list)
for item in dataset:
    stories[item["story_num"]].append(item["sent"])

story_nums, texts, series = [], [], []
for num, sents in stories.items():
    text = " ".join(sents)
    story_nums.append(num)
    texts.append(text)
    series.append(detect_series(text))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, convert_to_tensor=False)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings).astype("float32"))

# Сохраняем
with open("data/story_texts.pkl", "wb") as f:
    pickle.dump((story_nums, texts, series), f)
faiss.write_index(index, "data/story_index.faiss")

print("Предобработка завершена")
