
import json
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import PyPDF2
import speech_recognition as sr

# Set Streamlit page configuration as the first Streamlit command
st.set_page_config(page_title="Personalized Learning Assistant", page_icon=":books:", layout="wide")

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize conversation history and generated text
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    st.session_state.generated_text = ""

# Function to add custom CSS
def add_custom_css():
    st.markdown("""
    <style>
    /* Title customization */
    h1 {
        color: #4CAF50; /* Green */
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
    }
    
    h2, h3, h4 {
        color: #3a7ca5; /* Blue */
    }
    
    /* Customize the input box */
    .stTextInput>div>div>input {
        border: 2px solid #4CAF50;
        color: #3a7ca5;
    }
    
    /* Customize buttons */
    div.stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px;
        font-size: 18px;
        font-family: Arial, sans-serif;
    }
    
    /* Sidebar customization */
    .css-1d391kg {
        background-color: #e8f4fa; /* Light blue */
    }
    
    /* Footer styling */
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# Add custom symbols
def render_symbols(symbol):
    st.markdown(f"<h1 style='text-align: center;'>{symbol}</h1>", unsafe_allow_html=True)

# Prompt Template
def get_prompt(conversation_history):
    messages = [("system", "You are a helpful assistant. Please respond to the user's queries.")]
    for user_msg, assistant_msg in conversation_history:
        messages.append(("user", f"Question: {user_msg}"))
        messages.append(("assistant", assistant_msg))
    return ChatPromptTemplate.from_messages(messages)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to recognize speech and convert it to text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Please say something... 🎤")
        audio = recognizer.listen(source)

        try:
            st.info("Recognizing... 🔍")
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text} ✅")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your speech. Please try again. ❌")
        except sr.RequestError:
            st.error("Could not request results from the speech recognition service. Please check your internet connection. 🌐")
    return ""

# Function to save dictionary data to a JSON file
def save_to_json_file(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Apply custom CSS and symbols
add_custom_css()
render_symbols("📚")

# Main title and subtitle
st.title('Your Personalized Learning Assistant 🧠')
st.subheader("Get answers to your queries from AI or from your PDF content! 🔍")

# Sidebar for user settings and options
with st.sidebar:
    st.header("User Options 🛠")
    uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")
    use_voice_input = st.checkbox("🎙 Use voice input", help="Record audio instead of typing your question.")
    generate_mcqs = st.checkbox("📝 Generate Multiple Choice Questions (MCQs)", help="Create MCQs from the response.")

# Main Content Section
st.markdown("### Ask any question, and I’ll help you out! 💬")
st.write("You can either type your query or use your voice to input it.")

# Extract PDF text
pdf_text = ""
if uploaded_file:
    with st.spinner("📄 Extracting text from the uploaded PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF text extracted successfully! 🟢")
        st.expander("🔎 View Extracted PDF Text").write(pdf_text)

# User input options
input_text = st.text_input("Enter your question below ⌨:")

# Voice input processing
if use_voice_input and st.button("🎤 Record and Submit"):
    input_text = recognize_speech()

# Ollama LLaMA2 LLM 
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

# When user enters a question
if input_text:
    # Search PDF text
    if pdf_text and input_text.lower() in pdf_text.lower():
        st.write("### Found in PDF: 📚")
        st.write(f"*Query:* {input_text} was found in the uploaded PDF.")
        st.session_state.generated_text = pdf_text
    else:
        # Add user input to conversation history
        st.session_state.conversation_history.append((input_text, ""))
        
        # Get the prompt with conversation history
        prompt = get_prompt(st.session_state.conversation_history)
        
        # Generate response
        chain = prompt | llm | output_parser
        assistant_response = chain.invoke({"question": input_text})  # Use invoke() instead of run()
        
        # Update conversation history with assistant response
        st.session_state.conversation_history[-1] = (input_text, assistant_response)
        
        # Store the generated text
        st.session_state.generated_text = assistant_response

    # Display the conversation
    st.write("### Conversation History 🗣")
    with st.expander("Expand to view the conversation history 📜", expanded=True):
        for user_msg, assistant_msg in st.session_state.conversation_history:
            st.markdown(f"*You:* {user_msg} 🤔")
            st.markdown(f"*Assistant:* {assistant_msg} 💬")
            st.markdown("---")

# Generate MCQs if the option is selected and there is generated text
if generate_mcqs and st.session_state.generated_text:
    with st.spinner("Generating MCQs... 📝"):
        mcq_prompt = f"""
        You are an expert quiz maker for technical fields. Let's think step by step and
create a quiz with 10 multiple-choice questions about the following concept/content: {st.session_state.generated_text}.

The format of the quiz would be Multiple-choice, Use <h2> for the questions and <ul> with <li> along with radio buttons for each options so the user can select one option.
Format the entire output as HTML code for a webpage.

- Questions:
    <Question1>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    <Question2>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    ....
- Answers:
    <Answer1>: <a|b|c|d>
    <Answer2>: <a|b|c|d>
    ....
    Example:
    - Questions:
    - 1. What is the time complexity of a binary search tree?
        a. O(n)
        b. O(log n)
        c. O(n^2)
        d. O(1)
    - Answers: 
        1. b

    Please return only the HTML code that includes all 10 MCQs and a submit button at last to submit all the answers of above mcqs.
    ....
"""

        # Proceed with generating the MCQs using the LLM and saving the HTML
        mcq_chain = ChatPromptTemplate.from_messages([("user", mcq_prompt)]) | llm | output_parser
        
        # Execute the mcq_chain to get the result
        generated_mcqs = mcq_chain.invoke({"input": mcq_prompt})  # Pass the required input

        # Save the generated MCQs to an HTML file
        with open('generated_mcqs.html', 'w') as file:
            file.write(generated_mcqs)

        # Display success message or any further processing
        st.success("MCQs have been generated and saved to 'generated_mcqs.html'. 📥")
        st.download_button('Download MCQs 📥', data=generated_mcqs, file_name="generated_mcqs.html", mime='text/html')