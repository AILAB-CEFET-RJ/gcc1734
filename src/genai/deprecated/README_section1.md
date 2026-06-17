# Text Generation with Hugging Face Transformers

This project demonstrates how to use a pre-trained language model (GPT-2) to generate text using Python and the Hugging Face `transformers` library.

## 🧠 Example Output

```

Prompt: Once upon a time in a small village
Output: Once upon a time in a small village, there was a curious fox who loved to explore the nearby forest...

````

---

## 🚀 Getting Started

These instructions assume you're using **VS Code** on Windows, macOS, or Linux.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

### 2. Create a Virtual Environment (optional but recommended)

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install transformers torch
```

---

## 📄 Running the Script

1. Create a Python script (e.g., `text_gen.py`) and paste the code below:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Once upon a time in a small village"
result = generator(prompt, max_length=50, num_return_sequences=1)

print(result[0]["generated_text"])
```

2. Run the script in the terminal:

```bash
python text_gen.py
```

The first time you run it, the GPT-2 model (\~500MB) will be downloaded automatically.

---

## ✅ Requirements

* Python 3.7+
* `transformers` library
* `torch` (PyTorch)

---

## 📚 Resources

* [Transformers Documentation](https://huggingface.co/docs/transformers)
* [GPT-2 Model Card](https://huggingface.co/gpt2)

---

## 📜 License

MIT License