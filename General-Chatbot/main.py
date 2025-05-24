import streamlit as st
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google.genai import types
import base64

def get_answer(prompt, history, file=False):
    fix_prompt = f"""You are helpful assistant. Your task is answer the question from user.
                    User: {prompt}

                    Use this history conversation if you need to look at previous conversation contexts:
                    {history}"""
    
    if file:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[
                types.Part.from_bytes(
                    data=file_content,
                    mime_type=uploaded_file.type,
                ),
                fix_prompt
            ]
        )
    else:
        response = client.models.generate_content(
            model="gemma-3-12b-it",
            contents=fix_prompt
        )
    return response

def get_answer_web(prompt, history):
    fix_prompt = f"""You are helpful assistant. Your task is answer the question from user.
                    User: {prompt}

                    Use this history conversation if you need to look at previous conversation contexts:
                    {history}"""
    
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )

    response = client.models.generate_content(
        model="gemma-3-12b-it",
        contents=fix_prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )
    return response

# Function to encode the local image file to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Custom CSS to set the background image
def set_background_image(image_path):
    encoded_image = get_base64_of_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image using a local file path
background_image_path = "assets/Background Image-Opacity 30%.png"  # Relative path to the image
set_background_image(background_image_path)

# Set up the Streamlit app
st.title("Chatbot App")
st.image("assets/camp_header.jpg")

homepage_text = '''**Welcome to the future of conversation!**

Our **AI chatbot**, powered by cutting-edge **Large Language Model (LLM)** technology, offers a seamless and intelligent interaction experience.
Whether you're seeking expert knowledge or real-time information from the web, our chatbot delivers accurate, context-aware responses. Engage in natural, human-like conversations and explore a world of information at your fingertips—all in one place.
Experience the perfect blend of AI intelligence and web-savvy capabilities, designed to enhance your productivity and curiosity.'''

with st.container(border=True):
    st.markdown(homepage_text)

# User input for API key
api_key = st.text_input("Enter your API Key:", type="password")

if api_key:
    client = genai.Client(api_key=api_key)
else:
    st.warning("Please enter your API key to start the chatbot.")

choose_mode = st.radio("Choose Chatbot Capability", ["default knowledge", "search on internet"])

if choose_mode == "default knowledge":
    uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

    if uploaded_file is not None:
        file_content = uploaded_file.read()
        st.write(f"Uploaded file: {uploaded_file.name}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Let's say: Hello World!"):
    messages_history = st.session_state.get("messages", [])[-10:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("AI"):
        if choose_mode == "default knowledge":
            if uploaded_file:
                response = get_answer(prompt, history, file=True)
            else:
                response = get_answer(prompt, history, file=False)
        else:
            response = get_answer_web(prompt, history)
        answer = response.text
        st.markdown(answer)
    st.session_state.messages.append({"role": "AI", "content": answer})

    with st.expander("**History Chat:**"):
        st.code(history)

    model = response.model_version
    prompt_tokens = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count
    with st.expander("**Usage Details:**"):
        st.code(f'input token : {prompt_tokens}\noutput token : {completion_tokens}\nmodel : {model}')

# Clear chat history function
def clear_chat():
    st.session_state.messages = []
    st.rerun()

# Button to clear chat history
if st.button("Clear Chat"):
    clear_chat()