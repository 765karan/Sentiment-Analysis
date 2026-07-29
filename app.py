import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("😊 AI Sentiment Analysis")
st.write("Analyze the sentiment of any text using Machine Learning.")

user_input = st.text_area(
    "Enter your review",
    placeholder="Example: The movie was fantastic and I loved it!",
    height=150
)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        confidence = model.predict_proba(input_vector).max()

        st.subheader("Prediction")

        if str(prediction).lower() == "positive" or prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.progress(float(confidence))

st.markdown("---")
st.caption("Built with Streamlit and Scikit-learn")
