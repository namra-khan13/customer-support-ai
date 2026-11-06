# SupportIQ — AI Customer Support Intelligence

A portfolio project that analyzes customer-support messages and demonstrates an end-to-end applied AI/ML workflow.

## Features

- Ticket category classification with TF-IDF + Logistic Regression
- Lightweight sentiment baseline
- Rule-based priority estimation
- Category-specific suggested response templates
- SQLite persistence
- Streamlit analytics dashboard
- CSV export

## Tech stack

Python, pandas, scikit-learn, TF-IDF, Logistic Regression, SQLite, Streamlit, Git.

## Project structure

```text
customer_support_ai/
├── app.py
├── data/
├── models/                 # model is generated on first use (git-ignored)
├── src/
│   ├── __init__.py
│   ├── analytics.py
│   ├── database.py
│   └── pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run locally

### 1. Install Python

Use Python 3.10 or newer.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

The app opens in your browser. Submit a sample ticket, analyze it, and save it to populate the dashboard.

## Example messages to test

- "I was charged twice and need a refund urgently."
- "The app crashes when I try to log in."
- "My package is delayed and tracking has not changed."
- "I received a damaged item and want to return it."
- "Thank you, your support team was very helpful."

## Push this project to GitHub

1. Create a new **public** GitHub repository named `customer-support-ai`.
2. Open a terminal in this project folder.
3. Run:

```bash
git init
git add .
git commit -m "Initial commit: SupportIQ AI customer support project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/customer-support-ai.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to authenticate, complete the GitHub sign-in flow.

## Important limitations and next improvements

This is a learning/portfolio starter, not a production-grade support system.

- The classifier is trained on a small illustrative dataset included in `src/pipeline.py`. Its confidence is not a calibrated probability and its predictions must not be treated as ground truth.
- Sentiment is a transparent keyword baseline, not a trained sentiment model.
- Priority uses simple rules. Validate and tune these rules with domain experts.
- Add a held-out test set and report precision, recall, F1-score and a confusion matrix before presenting model performance.
- For a stronger version, collect a representative, permissioned dataset; add multilingual NLP, a retrieval-augmented knowledge base, authentication, audit logs, monitoring and human approval.
- Never commit private customer data, API keys, passwords, or secrets to GitHub.

## Resume description

**SupportIQ — AI Customer Support Intelligence System**  
Built a Streamlit application that classifies customer-support tickets using TF-IDF and Logistic Regression, estimates sentiment and urgency, stores analyses in SQLite, and visualizes support trends. Implemented response templates and CSV export to demonstrate an end-to-end applied machine-learning workflow.
