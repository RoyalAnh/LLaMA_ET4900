from llmtuner import create_web_demo
import os
from huggingface_hub import login
from dotenv import load_dotenv

'''
# Load biến môi trường từ file .env (nếu có)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
else:
    print("❌ Không tìm thấy Hugging Face token trong biến môi trường hoặc file .env") '''

def main():
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
