"""
app_streamlit.py — a simple visual demo so people don't need to use the API
directly. 
HOW TO RUN THIS FILE:
    streamlit run app_streamlit.py

It will open a page in your browser automatically.
"""
import streamlit as st

from router import predict

st.set_page_config(page_title="Phishing Detector", page_icon="🎣")

st.title("🎣 Phishing & Spam Detector")
st.write(
    "Paste a URL or an email below. This checks it using two machine "
    "learning models: one trained on URL structure, one trained on email text."
)

user_input = st.text_area("URL or email text", height=200, placeholder="Paste here...")

if st.button("Check", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a URL or some email text first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(user_input)

        verdict = result["verdict"]
        score = result["final_score"]

        if verdict == "phishing":
            st.error(f"⚠️ Likely PHISHING — confidence {score:.1%}")
        else:
            st.success(f"✅ Looks legitimate — confidence {(1 - score):.1%}")

        st.subheader("What the models saw")
        st.write(f"**Detected input type:** {result['input_type']}")

        if result["nlp_score"] is not None:
            st.write(f"**Email text model score:** {result['nlp_score']:.1%} phishing-like")

        if result["url_scores"]:
            st.write("**URL(s) checked:**")
            for item in result["url_scores"]:
                st.write(f"- `{item['url']}` → {item['score']:.1%} phishing-like")

        with st.expander("Raw result (for debugging)"):
            st.json(result)
