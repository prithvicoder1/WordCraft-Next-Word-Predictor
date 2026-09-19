# WordCraft ✦

**A simple studio for finding your next word.** Enter the beginning of a sentence and WordCraft suggests three possible next words using a trained LSTM language model.

[**Try the live app →**](https://wordcraft-next-word-predictor.onrender.com/) · [View the source](https://github.com/prithvicoder1/WordCraft-Next-Word-Predictor)

## What it does

- Accepts a short text prompt in a clean Streamlit interface.
- Shows the most likely next word alongside two alternatives.
- Uses the included model and tokenizer, so no training is needed to run the app.

Predictions depend on the training data and vocabulary. They can be surprising or imperfect, and text with no recognized words may not produce a suggestion.

## Run locally

You need **Python 3.12**. Clone the repository, install the dependencies, and start Streamlit:

```bash
git clone https://github.com/prithvicoder1/WordCraft-Next-Word-Predictor.git
cd WordCraft-Next-Word-Predictor
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address shown by Streamlit (usually **http://localhost:8501**). The repository already includes `lstm_model (1).h5`, `tokenizer.pkl`, and `max_len.pkl`; keep these files beside `app.py`.

## Run with Docker

```bash
docker build -f docker/Dockerfile -t wordcraft .
docker run --rm -p 8501:8501 wordcraft
```

Then open **http://localhost:8501**. If that port is busy, use `-p 18501:8501` and open **http://localhost:18501**.

## How it works

1. The tokenizer turns your text into a sequence of word IDs.
2. The sequence is padded to the length expected by the LSTM model.
3. The model scores possible next words; WordCraft displays the top three words available in the tokenizer's vocabulary.

The inference code lives in [`app.py`](app.py). The training material is in [`codefile.ipynb`](codefile.ipynb), with [`qoute_dataset.csv`](qoute_dataset.csv) as the included dataset. [`RNNimplementation.ipynb`](RNNimplementation.ipynb) explores a separate RNN implementation.

## Deploy on Render

Create a **Web Service** from this repository using **Docker**. Set the Dockerfile path to `docker/Dockerfile` and the health check path to `/_stcore/health`. The Dockerfile runs Streamlit on Render's `PORT` when provided.

## Credits

The training notebooks, dataset, and original model are based on [AkarshVyas/Next_word_prediction](https://github.com/AkarshVyas/Next_word_prediction).
