# Sci‑Fi Series Q\&A Bot

Этот репозиторий — готовый стек для локального чат‑бота, который отвечает на вопросы по популярным научно‑фантастическим сериалам. 

---

## Возможности

* **Локальная LLaMA 3** через [Ollama](https://ollama.com) (CPU‑friendly).
* Векторный поиск FAISS + Sentence‑Transformers для поиска релевантных сюжетов.
* **Telegram‑бот** для общения.
* **REST API** (FastAPI) + минимальный **HTML‑фронт**.
*  Полностью оффлайн: все данные хранятся и обрабатываются на вашем сервере.

---

## Структура проекта

```
series_bot/
├── data/                  # Индекс FAISS и сериализованные тексты
├── index.html             # Фронтенд 
├── preprocess.py          # Подготовка датасета → data/
├── model.py               # Запрос к Ollama API
├── bot.py                 # Telegram‑бот
├── server.py              # FastAPI /ask endpoint
├── requirements.txt       # Python‑зависимости
└── README.md              # этот файл
```

---

## Требования

| Компонент             | Версия / Примечание  |
| --------------------- | -------------------- |
| Python                | 3.8 +                |
| Ollama                | ≥ 0.1.32             |
| LLaMA 3 (8B Instruct) | Загружена в Ollama   |
| RAM                   | ≥ 8 ГБ (лучше 16 ГБ) |

> **CPU‑only**: видеокарта не требуется.

---

## Установка

```bash
# 1. Клонируем репо и создаём виртуальное окружение
python -m venv venv && source venv/bin/activate

# 2. Ставим зависимости
pip install -r requirements.txt

# 3. Устанавливаем Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 4. Подгружаем модель и прогреваем
ollama run llama3  # первый запуск скачает модель
```

---

## Подготовка данных

```bash
python preprocess.py   # ~3‑5 минут, создаст data/*
```

---

## Запуск компонентов

<details>
<summary>Бэкенд (REST API)</summary>

```bash
uvicorn server:app --host 0.0.0.0 --port 80
```

Swagger UI → `http://<IP>/docs`

</details>

<details>
<summary>Фронтенд</summary>

```bash
python -m http.server 8080 &
```

Затем откройте `http://<IP>:8080` в браузере.

</details>

<details>
<summary>Telegram‑бот</summary>

1. Создайте токен у @BotFather
2. Запишите его в `config.py` → `TELEGRAM_TOKEN = "123456:ABC..."`
3. Запустите:

```bash
python bot.py
```

</details>

---

## Сервис через systemd

> Пример юнита для FastAPI (аналогично задаются `bot.service`, `frontend.service`).

```ini
# /etc/systemd/system/sci-fi-api.service
[Unit]
Description=Sci‑Fi API
After=network.target

[Service]
User=root
WorkingDirectory=/root/series_bot
ExecStart=/root/series_bot/venv/bin/uvicorn server:app --host 0.0.0.0 --port 80
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sci-fi-api
```

---

## Поток обработки запроса

1. Пользователь задаёт вопрос (Telegram или REST).
2. **detect\_series()** определяет сериал по ключевым словам.
3. FAISS ищет 20 ближайших эпизодов, фильтрует по сериалу, выбирает топ‑3.
4. Каждая история обрезается до 1500 символов и собирается в `context`.
5. `model.py` отправляет `context + question` в Ollama → LLaMA 3.
6. Ответ возвращается пользователю.

---

## Troubleshooting

| Проблема                                     | Решение                                                            |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `405 OPTIONS /ask`                           | Проверьте CORS‑middleware в `server.py`.                           |
| `422 Unprocessable Entity` в `POST /ask`     | Фронт должен отправлять JSON `{"question": "..."}`.                |
| Telegram `BadRequest: message text is empty` | LLaMA вернула пустую строку. Проверьте логи Ollama и длину prompt. |
| `timeout` от Ollama                          | Увеличьте `timeout` в `model.py` или уменьшите длину контекста.    |


