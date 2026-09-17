FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY ["app.py", "tokenizer.pkl", "max_len.pkl", "lstm_model (1).h5", "./"]

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

CMD ["sh", "-c", "exec python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
