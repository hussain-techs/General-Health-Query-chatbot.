# 🏥 HealthAssist - General Health Query Chatbot

A conversational AI chatbot that answers general health-related questions
using Mistral-7B-Instruct model via Hugging Face Inference API.

## ✨ Features

- 💬 Interactive conversational interface
- 🛡️ Built-in safety filters for harmful content
- 🚨 Emergency symptom detection
- 📝 Prompt engineering for friendly, clear responses
- ⚠️ Automatic medical disclaimer on every response

## 🛠️ Tech Stack

- Python 3.8+
- Mistral-7B-Instruct (via Hugging Face API)
- Prompt Engineering for safe medical responses

## ⚙️ Setup

1. Clone the repository
   git clone https://github.com/YOUR_USERNAME/health-chatbot.git
   cd health-chatbot

2. Install dependencies
   pip install -r requirements.txt

3. Create .env file and add your Hugging Face token
   HF_TOKEN=your_huggingface_token_here

   Get free token at: https://huggingface.co/settings/tokens

## 🚀 Usage

Interactive mode:
   python health_chatbot.py

Demo mode (runs example queries):
   python health_chatbot.py --demo

## 💬 Example Queries

- "What causes a sore throat?"
- "Is paracetamol safe for children?"
- "How much water should I drink daily?"
- "What are the symptoms of type 2 diabetes?"

## 🛡️ Safety Features

- Harmful content filter — detects self-harm related queries
- Emergency detection — identifies emergency symptoms
- Medical disclaimer — added automatically to every response
- No diagnosis — never provides personal medical diagnosis

## ⚠️ Disclaimer

This chatbot provides GENERAL health information only.
It is NOT a substitute for professional medical advice,
diagnosis, or treatment. Always consult a qualified
healthcare professional for personal medical concerns.

## 📄 License

MIT License — free to use and modify.
