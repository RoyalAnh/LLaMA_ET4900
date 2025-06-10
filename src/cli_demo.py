# import readline
from llmtuner import ChatModel
from huggingface_hub import login
from dotenv import load_dotenv
import os
from pathlib import Path

def main():
    '''
    # Lấy đường dẫn đến thư mục gốc chứa file .env
    base_dir = Path(__file__).resolve().parent.parent
    dotenv_path = base_dir / ".env"

    # Load biến môi trường từ file .env ở thư mục gốc
    load_dotenv(dotenv_path=dotenv_path)

    # Lấy token từ biến môi trường
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        print("❌ Không tìm thấy Hugging Face token trong .env.")
        return

    try:
        login(token=hf_token)
        print("✅ Đã đăng nhập Hugging Face thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi đăng nhập: {e}")
        return
    '''
    chat_model = ChatModel()
    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            print("History has been removed.")
            continue

        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(query, history):
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]


if __name__ == "__main__":
    main()
