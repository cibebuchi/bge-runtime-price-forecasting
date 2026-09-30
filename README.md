# BGE Runtime-Constrained Price Forecasting

Interactive Streamlit research demonstration for Day-Ahead and Real-Time electricity price forecasting in PJM's BGE zone.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Historical model evaluation uses the packaged feature stores and does not require an API key. The optional experimental live refresh requires a PJM API key entered in the app sidebar.

## Scope

The application is designed for research visualization and reproducibility. The DA–RT spread display is a forecast diagnostic only and is not a financial or market recommendation.

Created by Chibuike C. Ibebuchi.
