import requests

def generate_answer(context, question):
    prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
            },
            timeout=300 
        )
        response.raise_for_status()
        data = response.json()

        output = data.get("response", "").strip()
        if not output:
            return "Модель не вернула ответ. Возможно, проблема с формулировкой запроса или слишком длинный контекст." 
        return output
    except Exception as e:
        return f"Ошибка при обращении к Ollama API: {e}"
