import streamlit as st
import pickle 

model=pickle.load(open("model.pkl", "rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))
st.title("Spam Email Classifier")
text=st.text_area("Enter email or message : ")
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text ")
    else:
        text_vec=vectorizer.transform([text])
        prediction=model.predict(text_vec)[0]

    if prediction==1:
        st.error("Spam")
    else :
        st.success("Not Spam")
        