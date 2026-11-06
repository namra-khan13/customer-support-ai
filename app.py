import streamlit as st
import pandas as pd
from src.pipeline import analyze_ticket
from src.database import init_db, save_ticket, load_tickets
from src.analytics import build_metrics

st.set_page_config(page_title="SupportIQ | AI Support Intelligence", page_icon="🤖", layout="wide")
init_db()

st.title("🤖 SupportIQ — AI Customer Support Intelligence")
st.caption("Classify support tickets, estimate sentiment and urgency, and draft a suggested response.")

with st.sidebar:
    st.header("About this project")
    st.write("A portfolio-ready demonstration combining NLP, machine learning, SQLite and business analytics.")
    st.info("Predictions are decision-support only. A human agent should review every suggested response and priority.")

tab1, tab2, tab3 = st.tabs(["Analyze a ticket", "Ticket analytics", "Project guide"])

with tab1:
    st.subheader("Submit a customer message")
    customer = st.text_input("Customer name (optional)")
    message = st.text_area("Customer message", height=150, placeholder="Example: I was charged twice for my order and need a refund.")
    if st.button("Analyze message", type="primary", disabled=not message.strip()):
        result = analyze_ticket(message)
        st.session_state["last_result"] = result
        st.session_state["last_message"] = message
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted category", result["category"].title())
        c2.metric("Sentiment", result["sentiment"].title())
        c3.metric("Priority", result["priority"].title())
        st.write("**Suggested response**")
        st.info(result["suggested_response"])
        st.caption(f"Category confidence: {result['category_confidence']:.0%} · Model: {result['model_name']}")
        if st.button("Save ticket to database"):
            save_ticket(customer, st.session_state["last_message"], result)
            st.success("Ticket saved.")

with tab2:
    st.subheader("Support operations overview")
    tickets = load_tickets()
    if tickets.empty:
        st.info("No saved tickets yet. Analyze and save a few sample messages to populate this dashboard.")
    else:
        metrics = build_metrics(tickets)
        a, b, c = st.columns(3)
        a.metric("Total tickets", metrics["total"])
        b.metric("High-priority tickets", metrics["high_priority"])
        c.metric("Negative sentiment", metrics["negative"])
        left, right = st.columns(2)
        with left:
            st.write("**Tickets by category**")
            st.bar_chart(tickets["category"].value_counts())
        with right:
            st.write("**Tickets by sentiment**")
            st.bar_chart(tickets["sentiment"].value_counts())
        st.write("**Recent tickets**")
        st.dataframe(tickets[["created_at", "customer", "message", "category", "sentiment", "priority"]], use_container_width=True)
        csv = tickets.to_csv(index=False).encode("utf-8")
        st.download_button("Download ticket history (CSV)", csv, "supportiq_tickets.csv", "text/csv")

with tab3:
    st.subheader("How this demo works")
    st.markdown("""
    1. **Text preprocessing:** lowercases and normalizes the submitted text.
    2. **Category prediction:** TF-IDF features + a trained scikit-learn classifier.
    3. **Sentiment analysis:** lightweight lexicon-based baseline for a transparent demo.
    4. **Priority estimation:** rules combine urgency phrases and negative sentiment.
    5. **Response suggestion:** category-specific editable response templates.
    6. **Persistence and reporting:** SQLite stores approved ticket analyses; Streamlit visualizes trends.

    This is a starter project, not a production customer-support system. Before production use, validate on representative,
    consented company data; measure per-class precision/recall; add authentication, access controls, monitoring and human review.
    """)
