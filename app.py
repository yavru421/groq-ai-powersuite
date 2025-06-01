import gradio as gr
import os
import json
import base64
import io
from PIL import Image
from text_extractors import extract_text_from_file as extract_text
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from cryptography.fernet import Fernet
from typing import Optional, List
import speech_recognition as sr

# Initialize Groq client
key = Fernet.generate_key()
cipher_suite = Fernet(key)
api_key_encrypted = None
groq_client = None

def initialize_groq_client(api_key: str):
    """Initialize Groq client with API key"""
    global groq_client
    try:
        groq_client = Groq(api_key=api_key)
        # Test connection
        groq_client.models.list()
        return True
    except Exception as e:
        groq_client = None
        print(f"Error initializing Groq client: {str(e)}")
        return False

def validate_api_key(api_key: str) -> bool:
    """Validate the Groq API key"""
    global api_key_encrypted
    try:
        os.environ['GROQ_API_KEY'] = api_key
        api_key_encrypted = cipher_suite.encrypt(api_key.encode())
        return initialize_groq_client(api_key)
    except Exception:
        return False

def ask_compound_beta(question, context="", model="compound-beta"):
    """Basic Q&A function using Compound-Beta model - COPIED FROM NOTEBOOK"""
    api_key = os.environ.get('GROQ_API_KEY')
    client = Groq(api_key=api_key)
    prompt = f"{context}\n{question}" if context else question
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model
    )
    return chat_completion.choices[0].message.content

def custom_groq_call(messages: List[ChatCompletionMessageParam], model="compound-beta"):
    """API playground function - COPIED FROM NOTEBOOK"""
    api_key = os.environ.get('GROQ_API_KEY')
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    return chat_completion.choices[0].message.content

def advanced_reasoning(prompt, model="compound-beta"):
    """Advanced reasoning and reflection function - COPIED FROM NOTEBOOK"""
    api_key = os.environ.get('GROQ_API_KEY')
    client = Groq(api_key=api_key)
    enhanced_prompt = f"Please provide thoughtful, reflective analysis for: {prompt}"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": enhanced_prompt}],
        model=model
    )
    return chat_completion.choices[0].message.content

def process_audio_file(audio_file_path):
    """Convert audio file to text using Groq Whisper API."""
    try:
        if not audio_file_path:
            return "No audio file provided."

        api_key = os.environ.get('GROQ_API_KEY')
        client = Groq(api_key=api_key)
        
        # Use Groq's audio transcription API (Whisper)
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file_path, file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        return transcription
    except Exception as e:
        return f"Error with audio transcription: {str(e)}"

def process_document(file_path, question="Summarize this document"):
    """Process document using Groq API."""
    try:
        # Extract text from document
        document_text = extract_text(file_path)
        
        # Use Groq API to analyze document
        api_key = os.environ.get('GROQ_API_KEY')
        client = Groq(api_key=api_key)
        
        prompt = f"Document content:\n{document_text}\n\nQuestion: {question}"
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="compound-beta"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing document with Groq API: {str(e)}"

def process_image(image_input, question="Analyze this image and describe what you see."):
    """Analyze image using Groq Vision API - expects PIL Image or numpy array only"""
    try:
        api_key = os.environ.get('GROQ_API_KEY')
        client = Groq(api_key=api_key)
        # Only handle PIL Image or numpy array
        if hasattr(image_input, 'save'):
            img_buffer = io.BytesIO()
            image_input.save(img_buffer, format='JPEG')
            image_bytes = img_buffer.getvalue()
        elif hasattr(image_input, 'shape'):
            pil_image = Image.fromarray(image_input)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG')
            image_bytes = img_buffer.getvalue()
        else:
            return "Unsupported image format"
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image with Groq API: {str(e)}"

def chat_response(message, history=None):
    """Handle chat completions using Groq API."""
    try:
        api_key = os.environ.get('GROQ_API_KEY')
        client = Groq(api_key=api_key)

        # Build conversation history
        messages = []
        if history:
            for turn in history:
                if isinstance(turn, list) and len(turn) == 2:
                    messages.append({"role": "user", "content": turn[0]})
                    messages.append({"role": "assistant", "content": turn[1]})
        
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            messages=messages,
            model="compound-beta"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Groq API chat completions: {str(e)}"

def fetch_models() -> list:
    """Fetch and return list of available models"""
    try:
        api_key = os.environ.get('GROQ_API_KEY')
        client = Groq(api_key=api_key)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        return [f"Error fetching models: {str(e)}"]

def get_vision_models(models: list) -> list:
    """Filter and return vision capable models"""
    vision_models = []
    for model in models:
        if "vision" in model.lower() or "image" in model.lower() or "scout" in model.lower():
            vision_models.append(model)
    return vision_models if vision_models else ["meta-llama/llama-4-scout-17b-16e-instruct"]

def intelligent_router(input_data):
    """Route user input to the appropriate functionality based on type and content."""
    if isinstance(input_data, str):
        return ask_compound_beta(input_data)
    elif hasattr(input_data, "name"):
        file_path = input_data.name
        if file_path.endswith(('.pdf', '.docx', '.pptx', '.txt', '.csv', '.json')):
            return process_document(file_path)
        else:
            return "Unsupported file type."
    elif hasattr(input_data, 'save') or hasattr(input_data, 'shape'):
        # It's a PIL Image or numpy array from Gradio
        return process_image(input_data)
    else:
        return "Unsupported input type."

def create_sidebar():
    """Create the sidebar configuration"""
    with gr.Column(scale=1):
        gr.Markdown("## Configuration")
        api_key_input = gr.Textbox(label="Enter Groq API Key", type="password")
        model_select = gr.Dropdown(
            label="Select Model", 
            choices=["compound-beta", "compound-beta-mini", "deepseek-r1-distill-llama-70b"], 
            value="compound-beta"
        )
        status_output = gr.Textbox(label="Status", interactive=False)

        def validate_and_initialize(api_key):
            if validate_api_key(api_key):
                models = fetch_models()
                return "API Key Validated!", gr.update(choices=models, visible=True)
            else:
                return "Invalid API Key", gr.update(visible=False)

        api_key_input.change(
            fn=validate_and_initialize,
            inputs=api_key_input,
            outputs=[status_output, model_select]
        )
        
        return api_key_input, model_select, status_output

def create_main_interface():
    """Create the main Gradio interface with modern UI design"""
    # Custom CSS for beautiful styling
    custom_css = """
    /* Force all Gradio text and input backgrounds to white and text to black */
    body, .gradio-container, .main-header, .main-header h1, .main-header p, .tab-nav, .tab-nav button, .tab-nav button.selected, .input-container, .output-container, .sidebar, .sidebar h2, button, .textbox, .dropdown, .file-upload, .chatbot, .message, .message.user, .message.bot, .feature-card, .emoji-icon, input, textarea, select, .gr-text-input, .gr-textbox, .gr-dropdown, .gr-file, .gradio-app, .gradio-row, .gradio-column {
        background: #fff !important;
        color: #111 !important;
        border-color: #bbb !important;
        box-shadow: none !important;
        text-shadow: none !important;
        -webkit-text-fill-color: #111 !important;
        caret-color: #111 !important;
    }
    /* Fix for dark mode browser overrides */
    html, body {
        background: #fff !important;
        color: #111 !important;
    }
    /* Ensure placeholder text is also visible */
    ::placeholder {
        color: #555 !important;
        opacity: 1 !important;
    }
    /* Remove any forced dark backgrounds from Gradio widgets */
    [class*="dark"], [class*="Dark"], [class*="-dark"], [class*="_dark"] {
        background: #fff !important;
        color: #111 !important;
    }
    """
    
    with gr.Blocks(title="🚀 Groq AI PowerSuite", css=custom_css) as demo:
        # Modern Header with glassmorphism effect
        with gr.Row(elem_classes="main-header"):
            gr.HTML("""
                <div style="text-align: center;">
                    <h1>🚀 Groq AI PowerSuite</h1>
                    <p>✨ Powered by Groq API - Experience the future of AI interaction</p>
                    <div style="margin-top: 1rem;">
                        <span class="status-indicator status-connected"></span>
                        <small style="color: rgba(255,255,255,0.8);">Ready for AI Magic</small>
                    </div>                </div>
            """)
        
        with gr.Row():
            with gr.Column(scale=4):
                # Modern tabbed interface with enhanced styling
                with gr.Tabs(elem_classes="tab-nav"):
                    # Text Q&A Tab with modern card design
                    with gr.Tab("💬 Intelligent Q&A", elem_id="qa-tab"):
                        with gr.Column(elem_classes="input-container"):
                            gr.HTML('<span class="emoji-icon">🧠</span><h3>Ask Anything with Compound-Beta</h3>')
                            question_input = gr.Textbox(
                                label="🤔 Your Question", 
                                lines=3, 
                                placeholder="What would you like to know? Ask me anything...",
                                elem_classes="textbox"
                            )
                            context_input = gr.Textbox(
                                label="📝 Optional Context", 
                                lines=2, 
                                placeholder="Provide additional context to get better answers...",
                                elem_classes="textbox"
                            )
                            qa_model_select = gr.Dropdown(
                                label="🤖 AI Model", 
                                choices=["compound-beta", "compound-beta-mini"], 
                                value="compound-beta",
                                elem_classes="dropdown"
                            )
                            ask_btn = gr.Button("✨ Get Answer", size="lg", variant="primary")
                        
                        with gr.Column(elem_classes="output-container"):
                            qa_output = gr.Textbox(
                                label="🎯 AI Response", 
                                lines=6, 
                                interactive=False,
                                placeholder="Your AI-powered answer will appear here..."
                            )
                        
                        def handle_qa(question, context, model):
                            if not question.strip():
                                return "Please enter a question first! 🤔"
                            return f"🚀 {ask_compound_beta(question, context, model)}"
                        
                        ask_btn.click(
                            fn=handle_qa,
                            inputs=[question_input, context_input, qa_model_select],
                            outputs=qa_output
                        )

                    # Document Q&A Tab with drag-and-drop styling
                    with gr.Tab("📄 Document Intelligence", elem_id="doc-tab"):
                        with gr.Column(elem_classes="input-container"):
                            gr.HTML('<span class="emoji-icon">📚</span><h3>Upload & Analyze Documents</h3>')
                            doc_upload = gr.File(
                                label="📎 Upload Document", 
                                file_types=[".pdf", ".docx", ".pptx", ".txt", ".csv", ".json"],
                                elem_classes="file-upload"
                            )
                            doc_question = gr.Textbox(
                                label="❓ Ask about your document", 
                                placeholder="What would you like to know about this document?",
                                elem_classes="textbox"
                            )
                            analyze_doc_btn = gr.Button("🔍 Analyze Document", size="lg", variant="primary")
                        
                        with gr.Column(elem_classes="output-container"):
                            doc_output = gr.Textbox(
                                label="📋 Document Analysis", 
                                lines=6, 
                                interactive=False,
                                placeholder="Upload a document and ask questions to see the analysis..."
                            )

                        def handle_document(file, question):
                            if file is None:
                                return "📁 Please upload a document first!"
                            if not question.strip():
                                question = "Summarize this document"
                            return f"📊 {process_document(file.name, question)}"

                        analyze_doc_btn.click(
                            fn=handle_document,
                            inputs=[doc_upload, doc_question],
                            outputs=doc_output
                        )

                    # Image Analysis Tab with visual enhancements
                    with gr.Tab("🖼️ Vision AI", elem_id="vision-tab"):
                        with gr.Column(elem_classes="input-container"):
                            gr.HTML('<span class="emoji-icon">👁️</span><h3>AI-Powered Image Analysis</h3>')
                            image_upload = gr.Image(
                                label="🖼️ Upload Image", 
                                type="pil",
                                elem_classes="image-upload"
                            )
                            image_question = gr.Textbox(
                                label="🔍 Ask about the image", 
                                placeholder="What do you want to know about this image?",
                                value="Analyze this image in detail",
                                elem_classes="textbox"
                            )
                            analyze_img_btn = gr.Button("👀 Analyze Image", size="lg", variant="primary")
                        
                        with gr.Column(elem_classes="output-container"):
                            image_output = gr.Textbox(
                                label="🎨 Vision Analysis", 
                                lines=6, 
                                interactive=False,
                                placeholder="Upload an image to see AI-powered visual analysis..."
                            )

                        def handle_image(image, question):
                            if image is None:
                                return "🖼️ Please upload an image first!"
                            if not question.strip():
                                question = "Analyze this image and describe what you see."
                            return f"👁️ {process_image(image, question)}"

                        analyze_img_btn.click(
                            fn=handle_image,
                            inputs=[image_upload, image_question],
                            outputs=image_output
                        )

                    # Audio Transcription Tab with waveform styling
                    with gr.Tab("🎤 Audio Intelligence", elem_id="audio-tab"):
                        with gr.Column(elem_classes="input-container"):
                            gr.HTML('<span class="emoji-icon">🎵</span><h3>Speech-to-Text with Groq Whisper</h3>')
                            audio_upload = gr.Audio(
                                label="🎙️ Upload Audio", 
                                type="filepath",
                                elem_classes="audio-upload"
                            )
                        
                        with gr.Column(elem_classes="output-container"):
                            audio_output = gr.Textbox(
                                label="📝 Transcription", 
                                lines=6, 
                                interactive=False,
                                placeholder="Upload audio to see AI-powered transcription..."
                            )

                        def handle_audio(audio_file):
                            if audio_file is None:
                                return "🎤 Please upload an audio file first!"
                            result = process_audio_file(audio_file)
                            return f"🎵 {result}"

                        audio_upload.change(
                            fn=handle_audio,
                            inputs=audio_upload,
                            outputs=audio_output
                        )

                    # Chat Assistant Tab with modern chat UI
                    with gr.Tab("💭 AI Companion", elem_id="chat-tab"):
                        with gr.Column(elem_classes="input-container"):
                            gr.HTML('<span class="emoji-icon">🤖</span><h3>Chat with Compound-Beta AI</h3>')                            chatbot = gr.Chatbot(
                                height=400,
                                elem_classes="chatbot"
                            )
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    label="💬 Message", 
                                    placeholder="Type your message here...",
                                    scale=4,
                                    elem_classes="textbox"
                                )
                                send_btn = gr.Button("📨 Send", scale=1, variant="primary")
                        
                        def respond(message, chat_history):
                            if not message.strip():
                                return "", chat_history
                            
                            response = chat_response(message, chat_history)
                            chat_history.append([f"🙋 {message}", f"🤖 {response}"])
                            return "", chat_history

                        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
                        send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])

                    # Advanced Reasoning Tab with thinking animations
                    with gr.Tab("🧠 Deep Reasoning", elem_id="reasoning-tab"):
                        with gr.Column(elem_classes="input-container"):
                            gr.HTML('<span class="emoji-icon">🤯</span><h3>Advanced AI Reasoning & Reflection</h3>')
                            reasoning_prompt = gr.Textbox(
                                label="🧐 Complex Question or Situation", 
                                lines=5,
                                placeholder="Describe a complex situation or philosophical question for deep analysis...",
                                elem_classes="textbox"
                            )
                            think_btn = gr.Button("🧠 Engage Deep Reasoning", size="lg", variant="primary")
                        
                        with gr.Column(elem_classes="output-container"):
                            reasoning_output = gr.Textbox(
                                label="💡 Deep Analysis", 
                                lines=8, 
                                interactive=False,
                                placeholder="Provide a complex question for thoughtful AI analysis..."
                            )

                        def handle_reasoning(prompt):
                            if not prompt.strip():
                                return "🤔 Please provide a question or situation to analyze!"
                            return f"💭 {advanced_reasoning(prompt)}"

                        think_btn.click(
                            fn=handle_reasoning,
                            inputs=reasoning_prompt,
                            outputs=reasoning_output
                        )

                    # API Playground Tab with code styling
                    with gr.Tab("🛠️ API Laboratory", elem_id="api-tab"):
                        with gr.Column(elem_classes="input-container"):
                            gr.HTML('<span class="emoji-icon">⚗️</span><h3>Direct API Access & Experimentation</h3>')
                            system_prompt = gr.Textbox(
                                label="⚙️ System Prompt (optional)", 
                                lines=2,
                                placeholder="Set the AI's behavior and context...",
                                elem_classes="textbox"
                            )
                            user_prompt = gr.Textbox(
                                label="👤 User Prompt", 
                                lines=5,
                                placeholder="Enter your direct API prompt...",
                                elem_classes="textbox"
                            )
                            playground_model = gr.Dropdown(
                                label="🤖 Model Selection", 
                                choices=["compound-beta", "compound-beta-mini"], 
                                value="compound-beta",
                                elem_classes="dropdown"
                            )
                            execute_btn = gr.Button("🚀 Execute API Call", size="lg", variant="primary")
                        
                        with gr.Column(elem_classes="output-container"):
                            playground_output = gr.Textbox(
                                label="📡 API Response", 
                                lines=8, 
                                interactive=False,
                                placeholder="API responses will appear here..."
                            )

                        def handle_playground(system, user, model):
                            if not user.strip():
                                return "💬 Please enter a user prompt!"
                            
                            messages = []
                            if system:
                                messages.append({"role": "system", "content": system})
                            messages.append({"role": "user", "content": user})
                            
                            result = custom_groq_call(messages, model)
                            return f"🤖 {result}"

                        execute_btn.click(
                            fn=handle_playground,
                            inputs=[system_prompt, user_prompt, playground_model],
                            outputs=playground_output
                        )

            # Sidebar
            with gr.Column(scale=1):
                create_sidebar()

    return demo

# Main function
# Create the Gradio interface
demo = create_main_interface()

# Launch for development if run directly
if __name__ == "__main__":
    demo.launch(debug=True)
else:
    # For Hugging Face Spaces
    demo.launch()
