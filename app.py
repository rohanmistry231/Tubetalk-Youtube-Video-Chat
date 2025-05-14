import os
import streamlit as st
from streamlit_option_menu import option_menu
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import re
import warnings
import logging
import uuid
from datetime import datetime
import io
import streamlit.components.v1 as components

# Configure logging to capture errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress all warnings and runtime errors for other libraries
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.CRITICAL)
logging.getLogger("youtube_transcript_api").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)

# Set page configuration once
st.set_page_config(
    page_title="TubeTalk - YouTube Video Chat",
    layout="wide",
    page_icon="🎥"
)

# Retrieve API token
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")
if not HF_API_TOKEN:
    st.error("Hugging Face API token not found. Please set HF_API_TOKEN in your secrets.toml file or Streamlit Cloud secrets. Get your token from https://huggingface.co/settings/tokens.")
    st.stop()

# Initialize models
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    try:
        client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_API_TOKEN)
    except Exception as e:
        st.error(f"Failed to initialize InferenceClient: {str(e)}")
        st.info("Ensure your HF_API_TOKEN is valid and you have access to the model on Hugging Face.")
        st.stop()
    return embedding_model, client

embedding_model, client = load_models()

# Custom embedding class with proper inheritance
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# Parse YouTube URL
def extract_video_id(url):
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&]+)',
        r'(?:https?://)?youtu\.be/([^?]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([^?]+)'
    ]
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            return match.group(1)
    return None

# Fetch transcript with multi-language support and detailed error logging
def get_transcript(video_id, preferred_language="en"):
    supported_languages = ["hi", "gu", "mr", "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    try:
        logger.info(f"Attempting to fetch transcript for video_id: {video_id} in language: {preferred_language}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[preferred_language])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        logger.info(f"Successfully fetched transcript in {preferred_language}")
        return transcript, transcript_list, preferred_language
    except TranscriptsDisabled as e:
        logger.error(f"Transcripts disabled for video_id: {video_id}. Error: {str(e)}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to fetch transcript in {preferred_language}. Error: {str(e)}")
        for lang in supported_languages:
            if lang != preferred_language:
                try:
                    logger.info(f"Falling back to language: {lang}")
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    transcript = " ".join(chunk["text"] for chunk in transcript_list)
                    logger.info(f"Successfully fetched transcript in {lang}")
                    return transcript, transcript_list, lang
                except Exception as inner_e:
                    logger.error(f"Failed to fetch transcript in {lang}. Error: {str(inner_e)}")
                    continue
        logger.error(f"No transcript available in any supported language for video_id: {video_id}")
        return None, None, None

# Process transcript
def process_transcript(transcript):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.create_documents([transcript])
        embeddings = HuggingFaceEmbeddings(embedding_model)
        texts = [doc.page_content for doc in chunks]
        embedded_docs = embeddings.embed_documents(texts)
        vector_store = FAISS.from_embeddings(
            [(text, embedding) for text, embedding in zip(texts, embedded_docs)],
            embeddings
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        st.error(f"Error processing transcript: {str(e)}")
        return None

# Call Hugging Face API with improved error handling
def generate_response_with_hf(context, question, api_token):
    generated_response = "Unable to generate response due to an unexpected error. Please try again later."
    try:
        if not context or not question:
            return "I don't have enough information to answer. Please ensure the transcript is available and the question is clear."

        client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_token)
        
        prompt = f"""
        You are a helpful assistant. Based on the following transcript context from a YouTube video: "{context}", answer the question: "{question}".
        Provide a concise response (50-100 words) in a clear and professional tone. Analyze the question to determine the requested format:
        - If the question specifies a format (e.g., "summarize in 2 lines," "list main points," "explain in detail"), follow that format exactly.
        - If no format is specified, provide a brief answer (1-2 sentences) followed by a section titled **Key Points** with 2-3 bullet points.
        Use markdown formatting for sections (e.g., **Key Points**) and bullet points (e.g., - Item). Ensure all sentences are complete and the tone is informative.
        """
        
        response = client.text_generation(prompt, max_new_tokens=150, temperature=0.7)
        generated_response = response.strip()
        
        if prompt in generated_response:
            generated_response = generated_response.split(prompt)[-1].strip()
        
        words = generated_response.split()
        if len(words) > 100:
            generated_response = " ".join(words[:100]) + "..."

        if generated_response and generated_response[-1] not in ".!?":
            last_punctuation = max(
                generated_response.rfind("."),
                generated_response.rfind("!"),
                generated_response.rfind("?")
            )
            if last_punctuation != -1:
                generated_response = generated_response[:last_punctuation + 1]

        return generated_response
    
    except Exception as e:
        error_message = "Unable to generate response. "
        if "rate limit" in str(e).lower():
            error_message += "Hugging Face API rate limit exceeded. Please wait and try again."
        elif "token" in str(e).lower() or "authentication" in str(e).lower():
            error_message += "Invalid or expired API token. Please check your HF_API_TOKEN in secrets.toml."
        elif "503" in str(e).lower() or "service unavailable" in str(e).lower():
            error_message += "Hugging Face API service is temporarily unavailable. Please try again later."
        else:
            error_message += f"An error occurred: {str(e)}. Please try again later."
        return error_message

# Parse and display response with improved robustness
def display_response(response):
    if not response:
        st.write("No response available.")
        return

    if not any(line.startswith("**") or line.startswith("- ") for line in response.split("\n")):
        st.write(response)
        return

    lines = [line.strip() for line in response.split("\n") if line.strip()]
    current_section = None
    section_items = []
    introduction = []

    for line in lines:
        if line.startswith("**") and line.endswith("**"):
            if current_section and section_items:
                st.subheader(current_section)
                for item in section_items:
                    st.write(f"• {item[2:]}")
            current_section = line.replace("**", "").strip()
            section_items = []
        elif line.startswith("- "):
            section_items.append(line)
        else:
            introduction.append(line)

    if introduction:
        st.write("\n".join(introduction))

    if current_section and section_items:
        st.subheader(current_section)
        for item in section_items:
            st.write(f"• {item[2:]}")

# Create retrieval chain
def create_chain(vector_store):
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        main_chain = (
            parallel_chain
            | RunnableLambda(lambda x: generate_response_with_hf(x['context'], x['question'], HF_API_TOKEN))
            | StrOutputParser()
        )
        return main_chain
    except Exception as e:
        logger.error(f"Error creating retrieval chain: {str(e)}")
        st.error(f"Error creating retrieval chain: {str(e)}")
        return None

# Format chat history as Markdown for download
def format_chat_history_as_markdown():
    if not st.session_state.chat_history:
        return "# Chat History\n\nNo chat history available."

    markdown_content = "# Chat History\n\n"
    for chat in st.session_state.chat_history:
        markdown_content += f"## Question (Asked on {chat['timestamp']})\n"
        markdown_content += f"{chat['question']}\n\n"
        markdown_content += f"## Answer\n"
        markdown_content += f"{chat['answer']}\n\n"
        markdown_content += "---\n\n"
    
    return markdown_content

# Reset transcript-related state on page load/refresh
def reset_transcript_state():
    st.session_state.transcript = None
    st.session_state.transcript_list = None
    st.session_state.vector_store = None
    st.session_state.video_id = None
    st.session_state.input_value = ""
    st.session_state.selected_language = "en"
    st.session_state.chat_history = []

# Home page
def home_page():
    st.title("Welcome to TubeTalk - YouTube Video Chat 🎥")
    st.markdown("""
    ### About This Application
    TubeTalk allows you to interact with YouTube videos by asking questions based on their transcripts in multiple languages, including Hindi, Gujarati, and Marathi. Powered by advanced AI models from Hugging Face and LangChain, the app processes video transcripts to provide concise, informative answers.

    **Key Features:**
    - View the YouTube video instantly by pasting its URL.
    - Ask questions about any YouTube video with captions in supported languages (e.g., Hindi, Gujarati, Marathi, English, etc.).
    - Receive smart responses tailored to your question's format (e.g., summaries, lists, or explanations).
    - View the full transcript and chat history within the app.
    - Download your chat history in Markdown format.
    - Secure integration with Hugging Face's Inference API.

    ### How to Use This Application
    1. **Navigate to the Chat Page**: Use the sidebar to select the "Chat" option.
    2. **Enter a YouTube URL**: Paste a valid YouTube video URL to instantly view the video.
    3. **Select a Language**: Choose your preferred transcript language from the dropdown.
    4. **Process the Transcript**: Click "Process Video" to fetch and analyze the transcript.
    5. **Ask Questions**: Type your question and click "Send" to get answers.
    6. **Review Transcript**: Expand the transcript section to view the full text.
    7. **Download Chat**: Use the "Download Chat" button to save your chat history as a Markdown file.

    ### Important Notes
    - Ensure the YouTube video has captions in at least one supported language for transcript processing.
    - A valid Hugging Face API token is required in your `secrets.toml` file.
    - For API token, visit [Hugging Face](https://huggingface.co/settings/tokens).

    ### Get Started
    Select the **Chat** page from the sidebar to start exploring YouTube videos!
    """)

# Chat page
def chat_page():
    st.title("Chat with YouTube Video")
    
    # Initialize session state with a flag to detect first run
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False

    # Reset transcript state on first run or page refresh
    if not st.session_state.app_initialized:
        reset_transcript_state()
        st.session_state.app_initialized = True

    # Debug mode toggle
    with st.sidebar:
        debug_mode = st.checkbox("Enable Debug Mode", value=False)

    # Input section with proper alignment
    st.header("Video Input")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        youtube_url = st.text_input("Enter YouTube URL", placeholder="e.g., https://www.youtube.com/watch?v=Gfr50f6ZBvo", key="youtube_url")
        # Extract video ID immediately when URL is entered
        video_id = extract_video_id(youtube_url) if youtube_url else None
        st.session_state.video_id = video_id
    with col2:
        language_options = {
            "hi": "Hindi",
            "gu": "Gujarati",
            "mr": "Marathi",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean"
        }
        selected_language = st.selectbox(
            "Select Transcript Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            key="language_select",
            index=list(language_options.keys()).index(st.session_state.selected_language)
        )
        st.session_state.selected_language = selected_language
    with col3:
        st.markdown(
            """
            <style>
            div.stButton > button {
                margin-top: 10px;
                vertical-align: middle;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        process_button = st.button("Process Video", key="process_button")

    # Video player section
    with st.container():
        st.header("Video Player")
        if st.session_state.video_id:
            embed_url = f"https://www.youtube.com/embed/{st.session_state.video_id}"
            components.iframe(embed_url, height=360, scrolling=False)
        else:
            st.info("No video loaded. Paste a valid YouTube URL to view the video here.")

    # Process transcript
    if process_button and youtube_url:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
            st.session_state.transcript = None
            st.session_state.transcript_list = None
            st.session_state.vector_store = None
            return
        
        with st.spinner("Fetching and processing transcript..."):
            transcript, transcript_list, used_language = get_transcript(video_id, preferred_language=st.session_state.selected_language)
            if transcript:
                st.session_state.transcript = transcript
                st.session_state.transcript_list = transcript_list
                st.session_state.vector_store = process_transcript(transcript)
                if st.session_state.vector_store is None:
                    st.session_state.transcript = None
                    st.session_state.transcript_list = None
                    st.error("Failed to process transcript. Please try again.")
                    return
                language_name = language_options.get(used_language, "Unknown")
                st.success(f"Transcript processed successfully in {language_name}!")
                st.session_state.chat_history = []
            else:
                st.error("Failed to fetch transcript. Check if the video has captions in the selected or any supported language. If this persists, check the app logs in Streamlit Cloud under 'Manage app' for more details.")
                st.session_state.transcript = None
                st.session_state.transcript_list = None
                st.session_state.vector_store = None

    # Transcript section
    with st.expander("View Transcript", expanded=False):
        transcript_key = f"transcript_area_{st.session_state.video_id or 'default'}"
        transcript_value = st.session_state.transcript if st.session_state.transcript else "No transcript available"
        st.text_area("Transcript", transcript_value, height=200, key=transcript_key)

    # Chat section
    st.header("Chat")
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            st.write(f"**You**: {chat['question']}")
            st.subheader("Answer")
            st.caption(f"Generated on {chat['timestamp']}")
            display_response(chat['answer'])
            st.divider()

    # Callback to update input value
    def update_input():
        st.session_state.input_value = st.session_state.question_input

    # Question input
    question = st.text_input(
        "Ask a question about the video",
        placeholder="e.g., What is this video about?",
        key="question_input",
        value=st.session_state.input_value,
        on_change=update_input
    )

    ask_button = st.button("Send", key="ask_button")

    # Buttons layout (Clear Chat and Download Chat) with adjusted spacing
    st.markdown(
        """
        <style>
        .button-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        div.stButton > button.clear-chat-button {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
        }
        div.stButton > button.clear-chat-button:hover {
            background-color: #e04343;
        }
        div.stButton > button.download-chat-button {
            background-color: #4b9bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
        }
        div.stButton > button.download-chat-button:hover {
            background-color: #437de0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use a container with flexbox to control button spacing
    with st.container():
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            clear_chat_button = st.button("Clear Chat", key="clear_chat_button", help="Clear the chat history and start fresh", type="primary")
        with col2:
            markdown_content = format_chat_history_as_markdown()
            buffer = io.StringIO()
            buffer.write(markdown_content)
            st.download_button(
                label="Download Chat",
                data=buffer.getvalue(),
                file_name="chat_history.md",
                mime="text/markdown",
                key="download_chat_button",
                help="Download the chat history as a Markdown file",
                type="primary"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Handle question
    if ask_button:
        if not question.strip():
            st.error("Please enter a question.")
        elif not st.session_state.vector_store:
            st.error("Please process a video transcript first.")
        else:
            with st.spinner("Generating answer..."):
                chain = create_chain(st.session_state.vector_store)
                if chain is None:
                    return
                if debug_mode:
                    st.write("Debug: Chain created successfully.")
                    st.write(f"Debug: Question received: {question}")
                answer = chain.invoke(question)
                if debug_mode:
                    st.write(f"Debug: Answer generated: {answer}")
                
                is_api_error = "rate limit" in answer.lower() or "service unavailable" in answer.lower() or "invalid or expired" in answer.lower()

                current_time = datetime.now().strftime("%I:%M %p IST, %A, %B %d, %Y")
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": current_time
                })
                
                st.write(f"**You**: {question}")
                st.subheader("Answer")
                st.caption(f"Generated on {current_time}")
                display_response(answer)
                st.divider()

                st.session_state.input_value = ""
                if not is_api_error:
                    st.rerun()

    # Handle Clear Chat button
    if clear_chat_button:
        st.session_state.chat_history = []
        st.rerun()

# Main app
def main():
    with st.sidebar:
        selected = option_menu(
            "TubeTalk - YouTube Video Chat",
            ["Home", "Chat"],
            menu_icon="film",
            icons=["house", "chat-dots"],
            default_index=0
        )

    if selected == "Home":
        home_page()
    elif selected == "Chat":
        chat_page()

if __name__ == "__main__":
    main()