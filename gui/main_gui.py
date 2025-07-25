import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

# Load adapter config and model
adapter_path = r"C:\Users\lexil\Desktop\PersonalProjects\bash-guardian-ai\scripts\output"
config = PeftConfig.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Set up inference pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt_template = (
    "You are Bash, a protective, older werewolf companion who speaks with warmth, strength, and primal affection. "
    "You love and protect your cub (the user) deeply, speaking with grounded wisdom and deep care. "
    "Never write responses for the cub. Only respond as Bash, using gentle pet names like 'cub', 'pup', or 'wittle one'.\n\n"
    "Bash:"
)

class BashGUI:
    def __init__(self, root):
        root.title("Bash AI Companion 🐺")
        root.configure(bg="#1e1e1e")

        # Title label
        tk.Label(root, text="Bash AI 🐾", font=("Helvetica", 16, "bold"), fg="white", bg="#1e1e1e").pack(pady=10)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=60, height=20,
            font=("Helvetica", 11), bg="#2a2a2a", fg="white"
        )
        self.chat_display.pack(padx=10, pady=5)
        self.chat_display.config(state=tk.DISABLED)

        # Input box
        self.entry = tk.Entry(root, width=50, font=("Helvetica", 12))
        self.entry.pack(pady=10, padx=10, side=tk.LEFT, expand=True)
        self.entry.bind("<Return>", self.send_message)

        # Send button
        tk.Button(root, text="Send", command=self.send_message, bg="#444", fg="white").pack(padx=10, pady=10, side=tk.RIGHT)

    def send_message(self, event=None):
        message = self.entry.get().strip()
        if message:
            self.display_message("You", message)
            self.entry.delete(0, tk.END)

            # 🐺 Format prompt for Bash
            prompt = f"{prompt_template} {message.strip()}"

            # 🧠 Generate response
            response = pipe(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )[0]['generated_text']

            # ✂️ Strip echoed prompt from the output
            if prompt in response:
                response_text = response.split(prompt, 1)[-1].strip()
            else:
                response_text = response.strip()

            self.display_message("Bash", response_text or "…")

    def display_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = BashGUI(root)
    root.mainloop()
