# 🎥 TubeTalk - YouTube Video Chat

**Interact with YouTube videos by watching them instantly and asking questions based on their transcripts in multiple languages!**

TubeTalk - YouTube Video Chat is an innovative web application that allows users to engage with YouTube videos by viewing them as soon as a URL is pasted and querying their transcripts in languages like Hindi, Gujarati, Marathi, English, and more. Powered by advanced AI technologies, including Hugging Face's language models and LangChain for retrieval-augmented generation, this app provides concise and accurate answers to user questions. Built with Streamlit, it offers a user-friendly interface for seamless interaction.

---

## ✨ Features

- **Instant Video Playback**: Paste a YouTube URL to instantly view the video in an embedded player.
- **Multi-Language Transcript Processing**: Fetch and analyze transcripts from YouTube videos in supported languages, including Hindi, Gujarati, Marathi, English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, and Korean.
- **Smart Q&A**: Ask questions about video content and receive tailored responses (e.g., summaries, lists, or detailed explanations) based on the transcript.
- **Interactive Chat Interface**: View chat history, clear conversations, and download them as Markdown files.
- **Transcript Viewer**: Access the full video transcript within the app after processing.
- **Debug Mode**: Enable debug logs for troubleshooting (optional).
- **Responsive Design**: Wide layout with a clean, modern UI optimized for desktop and mobile.
- **Secure API Integration**: Uses Hugging Face's Inference API with token-based authentication.

---

## 🛠️ Technologies Used

- **Frontend**: Streamlit for the web interface.
- **Backend**:
  - **YouTube Transcript API**: Fetches video transcripts in multiple languages.
  - **LangChain**: Handles text splitting, vector storage (FAISS), and retrieval-augmented generation.
  - **Hugging Face**:
    - `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
    - `mistralai/Mixtral-8x7B-Instruct-v0.1` for response generation.
- **Other Libraries**:
  - `sentence-transformers` for text embeddings.
  - `huggingface_hub` for API interactions.
  - `FAISS` for efficient vector search.
  - `re`, `datetime`, `io` for utility functions.
  - `streamlit.components` for embedding the YouTube video player.

---

## 🚀 Getting Started

Follow these steps to set up and run TubeTalk - YouTube Video Chat locally.

### Prerequisites

- Python 3.8 or higher
- A Hugging Face account with an API token
- A YouTube video with captions in a supported language (e.g., Hindi, Gujarati, Marathi, English, etc.)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/Tubetalk-Youtube-Video-Chat.git
   cd Tubetalk-Youtube-Video-Chat
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure your `requirements.txt` includes:

   ```
   streamlit
   streamlit-option-menu
   youtube-transcript-api
   langchain
   langchain-community
   sentence-transformers
   huggingface_hub
   faiss-cpu
   ```

4. **Set Up Hugging Face API Token**

   Create a `secrets.toml` file in the `.streamlit` directory (or directly in the project root for local runs):

   ```toml
   HF_API_TOKEN = "your-hugging-face-api-token"
   ```

   To get your token:
   - Visit [Hugging Face](https://huggingface.co/settings/tokens).
   - Generate a new token with `read` access.

5. **Run the Application**

   ```bash
   streamlit run app.py
   ```

   The app will open in your default browser at `http://localhost:8501`.

---

## 📖 Usage

1. **Navigate to the Chat Page**:
   - Use the sidebar to select "Chat."
2. **Enter a YouTube URL**:
   - Paste a valid YouTube video URL (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`) to instantly view the video in the embedded player.
   - Ensure the video has captions in a supported language for transcript processing.
3. **Select a Language**:
   - Choose your preferred transcript language (e.g., Hindi, Gujarati, Marathi, English) from the dropdown.
4. **Process the Transcript**:
   - Click "Process Video" to fetch and analyze the transcript. If the selected language is unavailable, the app will attempt other supported languages.
5. **Ask Questions**:
   - Type a question (e.g., "What is the main topic of the video?").
   - Click "Send" to get a response.
6. **View Transcript**:
   - Expand the "View Transcript" section to see the full transcript after processing.
7. **Manage Chat**:
   - **Clear Chat**: Reset the chat history.
   - **Download Chat**: Save the chat history as a Markdown file.

---

## ⚙️ How It Works

1. **Video Playback**:
   - The app extracts the video ID from the YouTube URL using regex patterns as soon as the URL is pasted.
   - The video is displayed instantly in an embedded player using Streamlit’s iframe component.
2. **Transcript Fetching**:
   - When "Process Video" is clicked, the `YouTubeTranscriptApi` fetches the transcript in the user’s preferred language (e.g., Hindi, Gujarati, Marathi) or falls back to other supported languages.
3. **Text Processing**:
   - The transcript is split into chunks using `RecursiveCharacterTextSplitter`.
   - Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
   - Embeddings are stored in a FAISS vector store for efficient retrieval.
4. **Question Answering**:
   - User questions are processed using a LangChain retrieval chain.
   - Relevant transcript chunks are retrieved and passed to the Hugging Face Inference API (`mistralai/Mixtral-8x7B-Instruct-v0.1`).
   - Responses are formatted in Markdown with sections like **Key Points**.
5. **UI Rendering**:
   - Streamlit renders the video player, transcript, chat history, and responses.
   - Custom CSS styles buttons for a polished look.

---

## 🐛 Troubleshooting

- **Invalid YouTube URL**:
  - Ensure the URL is correct and includes the video ID (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`).
  - If the video doesn’t display, verify the URL format and try again.
- **No Transcript Available**:
  - Verify the video has captions in a supported language (e.g., Hindi, Gujarati, Marathi, English).
- **API Errors**:
  - **Rate Limit Exceeded**: Wait and try again, or check your Hugging Face plan.
  - **Invalid Token**: Ensure `HF_API_TOKEN` is correct in `secrets.toml`.
  - **Service Unavailable**: Hugging Face servers may be down; try later.
- **Processing Errors**:
  - Enable "Debug Mode" in the sidebar to view detailed logs.
  - Check console output for specific error messages.
- **Language Issues**:
  - If the selected language is unavailable, the app will attempt other languages. Ensure captions exist for the video.

---

## 🌟 Future Enhancements

- Integration with additional LLMs for improved response generation in Indian languages.
- Enhanced UI with themes and accessibility features.
- Real-time video summary generation.
- Support for live YouTube streams with captions.
- Improved embedding models for better handling of Hindi, Gujarati, and Marathi.
- Video playback controls synced with transcript timestamps.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

---

## 📬 Contact

For questions or feedback, reach out via:
- **Email**: rohanmistry231@gmail.com