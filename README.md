# wordcraft-next-word-predictor

An LSTM next-word prediction app built with Streamlit and Docker.

## Run locally

Install Python 3.12, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app reads `lstm_model (1).h5`, `tokenizer.pkl`, and `max_len.pkl` from the project folder.

## Run with Docker

```bash
docker build -f docker/Dockerfile -t wordcraft-next-word-predictor .
docker run --rm -p 8501:8501 wordcraft-next-word-predictor
```

Open http://localhost:8501. Use another host port if 8501 is already in use, for example `-p 18501:8501`.

## Deploy on Render

Create a Web Service from this repository, select **Docker**, set the Dockerfile path to `docker/Dockerfile`, and set the health check path to `/_stcore/health`.

The training notebooks, dataset, and original model are based on [AkarshVyas/Next_word_prediction](https://github.com/AkarshVyas/Next_word_prediction).
