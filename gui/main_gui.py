import tkinter as tk
from tkinter import scrolledtext

class BashGUI:
    def __init__(self, root):
        root.title("Bash AI Companion 🐺")
        root.configure(bg="#1e1e1e")

        # Title label
        tk.Label(root, text="Bash AI 🐾", font=("Helvetica", 16, "bold"), fg="white", bg="#1e1e1e").pack(pady=10)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Helvetica", 11), bg="#2a2a2a", fg="white")
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
            # Placeholder AI response
            self.display_message("Bash", "Mmh, I hear you, cub...")

    def display_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = BashGUI(root)
    root.mainloop()
