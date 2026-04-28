import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ------------------------------------------
# IMPORTANT: change this path if needed
# ------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MODEL = "llama3.2"


# ------------------------------------------
# 1. Extract text from readable PDF
# ------------------------------------------
def extract_text_normal(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ------------------------------------------
# 2. Extract text from scanned PDF (OCR)
# ------------------------------------------
def extract_text_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        print(f"  OCR processing page {page_num + 1}/{len(doc)}...")
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    doc.close()
    return text


# ------------------------------------------
# 3. Smart extractor (auto detect)
# ------------------------------------------
def extract_text(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: '{pdf_path}'\n"
            f"Make sure the file is in: {os.path.abspath(os.path.dirname(pdf_path) or '.')}"
        )

    print("Trying normal text extraction...")
    text = extract_text_normal(pdf_path)

    if len(text.strip()) < 100:
        print("Scanned PDF detected. Using OCR (this may take a few minutes)...")
        text = extract_text_ocr(pdf_path)

    if len(text.strip()) == 0:
        raise ValueError("Could not extract any text from the PDF.")

    print(f"Extracted {len(text)} characters from PDF.")
    return text


# ------------------------------------------
# 4. Split text into chunks
# ------------------------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# ------------------------------------------
# 5. Create vector DB
# ------------------------------------------
def create_vector_db(chunks):
    print("Loading embedding model (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_texts(chunks, embeddings)
    print("Vector database created successfully.")
    return db


# ------------------------------------------
# 6. Ask question (RAG Q&A)
# ------------------------------------------
def ask_question(question, db):
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a study assistant. You ONLY answer using the context below.
If the answer is not explicitly found in the context, respond with:
"This information is not in the uploaded document."
DO NOT use your general knowledge. DO NOT make anything up.

Context from document:
{context}

Student Question: {question}

Answer strictly from the context above:"""



    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {e}\nMake sure Ollama is running: ollama serve"


# ------------------------------------------
# 7. Generate Summary
# ------------------------------------------
def generate_summary(text):
    sample = text[:3000]

    prompt = f"""You are a study assistant. Read the following document and provide:
1. A 3-5 sentence overall summary
2. The 5 most important key points as bullet points

Document:
{sample}

Provide the summary now:"""

    try:
        print("\nGenerating summary (this may take a moment)...")
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {e}\nMake sure Ollama is running: ollama serve"


# ------------------------------------------
# 8. Generate Quiz (MCQs)
# ------------------------------------------
def generate_quiz(text, num_questions=5):
    sample = text[:3000]

    prompt = f"""You are a quiz generator. Based on the document below, generate exactly {num_questions} multiple choice questions.

Format each question EXACTLY like this:
Q1. [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [Correct letter]

Document:
{sample}

Generate {num_questions} MCQs now:"""

    try:
        print(f"\nGenerating {num_questions} quiz questions (this may take a moment)...")
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {e}\nMake sure Ollama is running: ollama serve"


# ------------------------------------------
# 9. Generate Flashcards
# ------------------------------------------
def generate_flashcards(text, num_cards=5):
    sample = text[:3000]

    prompt = f"""You are a flashcard generator. Based on the document below, generate exactly {num_cards} flashcards.

Format EXACTLY like this:
CARD 1
Front: [Term or concept]
Back: [Definition or explanation]

CARD 2
Front: [Term or concept]
Back: [Definition or explanation]

Document:
{sample}

Generate {num_cards} flashcards now:"""

    try:
        print(f"\nGenerating {num_cards} flashcards (this may take a moment)...")
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {e}\nMake sure Ollama is running: ollama serve"


# ------------------------------------------
# 10. Interactive Quiz Mode
# ------------------------------------------
def run_interactive_quiz(text, num_questions=5):
    quiz_raw = generate_quiz(text, num_questions)

    print("\n" + "="*50)
    print("              QUIZ TIME!")
    print("="*50)
    print(quiz_raw)
    print("\n" + "-"*50)
    print("Review the answers above.")


# ------------------------------------------
# 11. Interactive Flashcard Mode
# ------------------------------------------
def run_interactive_flashcards(text, num_cards=5):
    cards_raw = generate_flashcards(text, num_cards)

    # Parse cards
    cards = []
    current_card = {}
    for line in cards_raw.split('\n'):
        line = line.strip()
        if line.lower().startswith('front:'):
            current_card['front'] = line[6:].strip()
        elif line.lower().startswith('back:'):
            current_card['back'] = line[5:].strip()
            if 'front' in current_card:
                cards.append(current_card)
                current_card = {}

    if not cards:
        print("\n" + "="*50)
        print(cards_raw)
        return

    print("\n" + "="*50)
    print("           FLASHCARD SESSION")
    print("="*50)
    print(f"Total cards: {len(cards)}")
    print("Press Enter to reveal answer | type 'skip' to skip | type 'quit' to stop\n")

    for i, card in enumerate(cards):
        print(f"\n--- Card {i+1} of {len(cards)} ---")
        print(f"FRONT: {card['front']}")
        user = input("Press Enter to reveal answer... ").strip().lower()
        if user == 'quit':
            break
        if user != 'skip':
            print(f"BACK:  {card['back']}")
        input("Press Enter for next card...")

    print("\nFlashcard session complete!")


# ------------------------------------------
# 12. Main Menu
# ------------------------------------------
def show_menu():
    print("\n" + "="*50)
    print("  STUDYBUDDY AI — Choose a feature:")
    print("="*50)
    print("  1.  Ask a Question  (Q&A)")
    print("  2.  Get a Summary")
    print("  3.  Take a Quiz     (MCQs)")
    print("  4.  Study Flashcards")
    print("  5.  Exit")
    print("-"*50)


# ------------------------------------------
# 13. MAIN PROGRAM
# ------------------------------------------
if __name__ == "__main__":

    # -----------------------------------------------
    # SET YOUR PDF PATH HERE
    # -----------------------------------------------
    pdf_path = "10th Physics Ch17 Extra MCQs.pdf"

    print("\n" + "="*50)
    print("  STUDYBUDDY AI — Document Intelligence")
    print("="*50)

    print("\nLoading PDF...")
    try:
        text = extract_text(pdf_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        exit(1)
    except ValueError as e:
        print(f"\nERROR: {e}")
        exit(1)

    print("\nPreparing chunks...")
    chunks = split_text(text)

    print("\nBuilding vector database...")
    db = create_vector_db(chunks)

    print("\nStudyBuddy AI is ready!")

    # Main loop
    while True:
        show_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            print("\n[Q&A Mode] Type 'back' to return to menu.\n")
            while True:
                question = input("Ask a question: ").strip()
                if not question:
                    continue
                if question.lower() == "back":
                    break
                print("\nThinking...")
                answer = ask_question(question, db)
                print(f"\nAnswer: {answer}")
                print("\n" + "-"*50)

        elif choice == "2":
            summary = generate_summary(text)
            print("\n" + "="*50)
            print("         DOCUMENT SUMMARY")
            print("="*50)
            print(summary)
            print("="*50)
            input("\nPress Enter to return to menu...")

        elif choice == "3":
            try:
                num = int(input("How many questions? (1-10, default 5): ").strip() or "5")
                num = max(1, min(10, num))
            except ValueError:
                num = 5
            run_interactive_quiz(text, num)
            input("\nPress Enter to return to menu...")

        elif choice == "4":
            try:
                num = int(input("How many flashcards? (1-10, default 5): ").strip() or "5")
                num = max(1, min(10, num))
            except ValueError:
                num = 5
            run_interactive_flashcards(text, num)
            input("\nPress Enter to return to menu...")

        elif choice == "5":
            print("\nGoodbye! Keep studying!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
