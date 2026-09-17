import pickle
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Wordcraft — Next Word Studio",
    page_icon="✦",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --background: #0b0e13;
        --panel: #151a22;
        --border: #303844;
        --text: #f4f5f2;
        --muted: #a5afba;
        --accent: #c6f36b;
    }

    .stApp {
        background:
            radial-gradient(circle at 50% -10%, #263526 0, transparent 42%),
            linear-gradient(#0b0e13, #0b0e13);
        color: var(--text);
        font-family: "DM Sans", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="block-container"] {
        max-width: 820px;
        padding-top: 3rem;
        padding-bottom: 5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 5rem;
        color: var(--text);
    }

    .brand-name {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -.04em;
    }

    .brand-name span {
        color: var(--accent);
    }

    .brand-tag {
        color: var(--muted);
        font-size: .75rem;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .eyebrow {
        color: var(--accent);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .22em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: clamp(3.4rem, 9vw, 6rem);
        font-weight: 700;
        line-height: .96;
        letter-spacing: -.085em;
        margin: 0 0 1.5rem;
        color: var(--text);
    }

    .hero-title em {
        color: var(--accent);
        font-style: normal;
    }

    .hero-copy {
        max-width: 560px;
        color: var(--muted);
        font-size: 1.08rem;
        line-height: 1.7;
        margin-bottom: 3rem;
    }

    .section-label {
        color: var(--muted);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .17em;
        text-transform: uppercase;
        margin-bottom: .8rem;
    }

    [data-testid="stTextArea"] textarea {
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        font-family: "DM Sans", sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px #c6f36b33;
    }

    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3.4rem;
        background: var(--accent);
        color: #11170b;
        border: 0;
        border-radius: 14px;
        font-family: "Space Grotesk", sans-serif;
        font-size: 1rem;
        font-weight: 700;
        margin-top: .7rem;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: #d8ff8d;
        color: #11170b;
        border: 0;
    }

    .result {
        margin-top: 2.5rem;
        padding: 2rem;
        background: linear-gradient(135deg, #202a22, #151a22 65%);
        border: 1px solid #46523c;
        border-radius: 22px;
    }

    .result-label {
        color: var(--accent);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .18em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .result-text {
        color: var(--text);
        font-family: "Space Grotesk", sans-serif;
        font-size: clamp(1.5rem, 5vw, 2.5rem);
        line-height: 1.3;
        letter-spacing: -.045em;
        overflow-wrap: anywhere;
    }

    .result-text strong {
        color: var(--accent);
    }

    .alternatives {
        color: var(--muted);
        font-size: .9rem;
        margin-top: 1.4rem;
    }

    .footer {
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.6;
        margin-top: 5rem;
        padding-top: 1.4rem;
    }

    @media (max-width: 600px) {
        [data-testid="block-container"] { padding-top: 1.5rem; }
        .brand { margin-bottom: 4rem; }
        .brand-tag { display: none; }
        .result { padding: 1.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    model = load_model(BASE_DIR / "lstm_model (1).h5", compile=False)

    with open(BASE_DIR / "tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open(BASE_DIR / "max_len.pkl", "rb") as file:
        max_len = pickle.load(file)

    return model, tokenizer, max_len


def predict_words(text, model, tokenizer, max_len):
    tokens = tokenizer.texts_to_sequences([text.lower()])[0]
    if not tokens:
        return []

    sequence = pad_sequences([tokens], maxlen=max_len, padding="pre")
    probabilities = model.predict(sequence, verbose=0)[0]

    suggestions = []
    for index in np.argsort(probabilities)[::-1]:
        word = tokenizer.index_word.get(int(index))
        if word:
            suggestions.append(word)
        if len(suggestions) == 3:
            break

    return suggestions


st.markdown(
    """
    <div class="brand">
        <div class="brand-name">✦ WORD<span>CRAFT</span></div>
        <div class="brand-tag">Next word studio · LSTM model</div>
    </div>
    <div class="eyebrow">A little glimpse into what comes next</div>
    <h1 class="hero-title">Find your<br><em>next word.</em></h1>
    <p class="hero-copy">
        Give your sentence a beginning. Our language model will imagine
        the word that could come next.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    st.markdown(
        '<div class="section-label">Your starting words</div>',
        unsafe_allow_html=True,
    )
    user_input = st.text_area(
        "Your starting words",
        placeholder="Try: the meaning of life",
        height=120,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Predict the next word  ↗")

if submitted:
    if not user_input.strip():
        st.warning("Type a few words first.")
    else:
        try:
            model, tokenizer, max_len = load_resources()
            suggestions = predict_words(
                user_input, model, tokenizer, max_len
            )

            if suggestions:
                original = escape(user_input.strip())
                best_word = escape(suggestions[0])
                alternatives = " · ".join(
                    escape(word) for word in suggestions[1:]
                )

                st.markdown(
                    f"""
                    <div class="result">
                        <div class="result-label">✦ Your next word</div>
                        <div class="result-text">
                            {original} <strong>{best_word}</strong>
                        </div>
                        <div class="alternatives">
                            Other possibilities: {alternatives}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "The model did not recognize these words. "
                    "Try a different sentence."
                )
        except Exception as error:
            st.error(f"Could not load the model: {error}")

st.markdown(
    """
    <div class="footer">
        WORDCRAFT · Predictions are generated by your trained LSTM model.
        They may be surprising, imperfect, or wonderfully unexpected.
    </div>
    """,
    unsafe_allow_html=True,
)