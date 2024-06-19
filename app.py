import streamlit as st
import pickle






vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))






def classify_email(text):   
    
    transformed_text = vectorizer.transform([text])
   
    prediction = model.predict(transformed_text)
    return prediction[0]

def main():
    st.title("Email spam classification")
    

    email_text = st.text_area("Enter the email  here")

    if st.button("Classify"):
        if email_text:
            result = classify_email(email_text)
            st.write(f"Prediction: {result}")
        else:
            st.write("Please enter the email text.")

if __name__ == '__main__':
    main()

