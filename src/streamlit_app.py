import os
from urllib.parse import urljoin

import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://titanic-dataset-2.onrender.com",
)


def call_predict(age: float, fare: float) -> dict:
    url = urljoin(BACKEND_URL.rstrip("/"), "/predict")
    response = requests.post(url, json={"age": age, "fare": fare}, timeout=10)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
    st.title("Titanic Survival Predictor")
    st.markdown(
        "Use the form below to predict whether a passenger would survive the Titanic disaster." 
        "The app calls the backend API for inference."
    )

    with st.sidebar:
        st.header("Backend settings")
        st.write("Using backend URL:")
        st.code(BACKEND_URL)
        if st.button("Check backend health"):
            try:
                health = requests.get(urljoin(BACKEND_URL.rstrip("/"), "/health"), timeout=10)
                health.raise_for_status()
                data = health.json()
                st.success(f"Backend healthy: {data}")
            except Exception as exc:
                st.error(f"Backend health check failed: {exc}")

    age = st.number_input("Passenger age", min_value=0.0, max_value=120.0, value=29.0)
    fare = st.number_input("Ticket fare", min_value=0.0, value=32.2)

    if st.button("Predict survival"):
        try:
            result = call_predict(age, fare)
            survived_label = "Yes" if result["survived"] else "No"
            st.metric(label="Would survive?", value=survived_label)
            st.write(f"Predicted probability of survival: {result['probability']:.3f}")
            st.json(result)
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")

    st.caption("Note: set BACKEND_URL on Streamlit service to your backend URL in Render.")


if __name__ == "__main__":
    main()
