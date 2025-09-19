BeaconMedicalAi AMA
BeaconMedicalAi AMA is a professional application for generating AMA-compliant impairment reports and rebuttals from medical and legal documents. It supports PDF, media, images, and text inputs, leveraging AI models and vectorstores for accurate analysis based on AMA Guides (4th or 6th Edition).
Features

Generate AMA impairment reports and rebuttals
Process PDFs, images, audio, video, and ZIP files
Secure user sessions with AES-256 encryption
Gradio-based UI for interactive report generation
Supports 4th and 6th Editions of AMA Guides
Media processing for audio/video transcription
Robust path resolution using sys and os.path

Installation

Clone the repository:git clone https://github.com/your-repo/beacon_medical_ai.git
cd beacon_medical_ai


Create a virtual environment and install dependencies:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Copy .env.example to .env and set your API keys:cp .env.example .env

Edit .env to include OPENAI_API_KEY, GROQ_API_KEY, and HF_API_KEY.
Run the application:./run_app.sh



Usage

Launch the app and log in with credentials (e.g., username: michael rice, password: 4321).
Upload files (PDF, ZIP, media) or input text.
Select AMA Guides edition (4th or 6th) and mode (Report or Rebuttal).
Generate reports or rebuttals, download results as PDFs, or interact via the chatbot.

Project Structure
beacon_medical_ai/
├── README.md
├── requirements.txt
├── .env.example
├── setup.py
├── run_app.sh
├── src/
│   ├── main.py
│   ├── auth/
│   │   └── authentication.py
│   ├── core/
│   │   ├── user_data.py
│   │   ├── vectorstore.py
│   │   └── models.py
│   ├── processors/
│   │   ├── file_processors.py
│   │   ├── text_processors.py
│   │   └── pdf_handlers.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── report_generator.py
│   │   ├── rebuttal_generator.py
│   │   └── combiner.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   ├── logging_utils.py
│   │   └── web_utils.py
│   └── ui/
│       ├── __init__.py
│       └── gradio_interface.py
└── templates/
    ├── ama_imp_prmpt1.txt
    └── ama_combine_impairment.txt

Requirements
See requirements.txt for dependencies. Key libraries include:

gradio
langchain
openai
groq
moviepy
markdown-pdf
Pillow
speechrecognition

Environment Variables
Set these in .env:

OPENAI_API_KEY: Your OpenAI API key
GROQ_API_KEY: Your Groq API key
HF_API_KEY: Your Hugging Face API key
PORT: Server port (default: 7860)

Contributing
Submit issues or PRs to the repository. Ensure code follows PEP 8, uses sys for path resolution, and includes tests.
License
MIT License