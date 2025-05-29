import streamlit as st
import google.generativeai as genai

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyBHok9oP0qreBFA7yBB3IWufsVJDHmGjzM"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def get_gemini_response(prompt):
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("Gemini AI Chatbot")
    
    initialize_session_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Chat with Gemini"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get Gemini response
        response = get_gemini_response(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
