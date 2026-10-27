
import os
import re
import hashlib
import streamlit as st
import fitz
from groq import Groq
from google import genai
from google.genai import types

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF Study AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_OCR_MODEL = "gemini-2.5-flash"

MAX_HISTORY_MESSAGES = 8
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 700

# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef5ff 100%);
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #172554, #2563eb);
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 36px;
}
.hero p {
    color: #e0e7ff;
    font-size: 16px;
}
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 12px;
}
.stButton button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# API KEYS
# =========================================================

def get_secret(name):
    """Read secrets from Streamlit Cloud or environment variables."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(name, "").strip()


# =========================================================
# PDF EXTRACTION AND GEMINI OCR
# =========================================================

def extract_pdf_text(uploaded_file, gemini_client):
    """
    Extract selectable text normally.
    If a page has little or no text, render it as an image and ask
    Gemini to perform visual text recognition.
    """

    pdf_bytes = uploaded_file.getvalue()
    extracted_pages = []
    ocr_pages = 0
    total_pages = 0

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            total_pages = len(pdf)

            if total_pages == 0:
                raise ValueError("The PDF contains no pages.")

            progress = st.progress(0)
            status = st.empty()

            for page_number, page in enumerate(pdf, start=1):
                status.info(
                    f"Reading page {page_number} of {total_pages}..."
                )

                # First try ordinary text extraction.
                page_text = page.get_text("text").strip()

                # If the page is scanned, use Gemini OCR.
                if len(page_text) < 40:
                    try:
                        # Render at moderate resolution to limit memory use.
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(1.35, 1.35),
                            alpha=False
                        )
                        image_bytes = pix.tobytes("png")

                        ocr_prompt = """
You are performing OCR on a page from a student's study PDF.

Transcribe all readable text on this page as accurately as possible.
Preserve headings, paragraphs, numbered lists, formulas, definitions,
tables, and labels where possible.

Do not summarize, explain, or answer questions.
Do not invent missing words. Mark genuinely unreadable text as [unclear].
Return only the recognized page text.
"""

                        response = gemini_client.models.generate_content(
                            model=GEMINI_OCR_MODEL,
                            contents=[
                                ocr_prompt,
                                types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type="image/png"
                                )
                            ]
                        )

                        recognized_text = (
                            response.text.strip()
                            if response.text
                            else ""
                        )

                        if recognized_text:
                            page_text = recognized_text
                            ocr_pages += 1

                    except Exception as exc:
                        status.warning(
                            f"OCR failed on page {page_number}. "
                            "Continuing with other pages."
                        )
                        # Keep any text extracted before OCR.
                        if not page_text:
                            page_text = (
                                f"[Text could not be extracted from "
                                f"page {page_number}.]"
                            )

                extracted_pages.append(
                    f"\n--- PAGE {page_number} ---\n{page_text}"
                )

                progress.progress(page_number / total_pages)

            progress.empty()
            status.empty()

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not open this PDF. It may be damaged, encrypted, "
            "or unsupported."
        ) from exc

    complete_text = "\n".join(extracted_pages).strip()

    if not complete_text or not any(
        len(page.strip()) > 40
        for page in extracted_pages
    ):
        raise ValueError(
            "No useful text could be extracted. Check that the PDF "
            "is readable and try again."
        )

    return complete_text, total_pages, ocr_pages


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            boundary = chunk.rfind("\n")
            if boundary > chunk_size * 0.65:
                end = start + boundary
                chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


# =========================================================
# GROQ CHAT COMPLETION
# =========================================================

def call_groq(client, system_prompt, user_prompt, history=None):
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.25,
        max_tokens=3500
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("Groq returned an empty response.")

    return answer


# =========================================================
# TOPIC EXTRACTION
# =========================================================

def build_topic_list(client, chunks):
    extracted_lists = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""
Identify the main topics and subtopics in this course material.

Include definitions, key terms, theories, models, processes, formulas,
classifications, examples, and comparisons when present.

SECTION {index} OF {len(chunks)}:
{chunk}

Return a concise bullet list of topic names.
Use only information supported by this section.
"""

        answer = call_groq(
            client,
            "You are a careful academic topic extractor.",
            prompt
        )

        extracted_lists.append(
            f"SECTION {index}\n{answer}"
        )

    combined = "\n\n".join(extracted_lists)

    # Merge lists in batches to avoid an excessively long final prompt.
    batch_size = 12
    merged_batches = []

    for start in range(0, len(extracted_lists), batch_size):
        batch = "\n\n".join(
            extracted_lists[start:start + batch_size]
        )

        merged = call_groq(
            client,
            "Organize course topics accurately. Never invent topics.",
            f"""
Merge the following topic lists.
Remove duplicates, retain important subtopics, and organize them
under clear headings. Return a numbered outline.

{batch}
"""
        )
        merged_batches.append(merged)

    if len(merged_batches) == 1:
        return merged_batches[0]

    return call_groq(
        client,
        "You are an accurate academic syllabus organizer.",
        f"""
Combine these topic outlines into one complete outline.
Remove duplicate topics but preserve distinct subtopics.
Use numbered headings and bullets. Do not invent topics.

{chr(10).join(merged_batches)[:50000]}
"""
    )


# =========================================================
# ASK QUESTIONS ABOUT THE PDF
# =========================================================

def answer_question(client, question, chunks, history=None):
    # Simple keyword-based retrieval of likely relevant sections.
    terms = set(
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9_+-]+", question)
        if len(word) > 2
    )

    ranked = []

    for index, chunk in enumerate(chunks):
        chunk_terms = set(
            word.lower()
            for word in re.findall(r"[a-zA-Z0-9_+-]+", chunk)
        )

        score = len(terms.intersection(chunk_terms))
        ranked.append((score, index, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)

    # Include the best matching sections in original PDF order.
    selected = sorted(ranked[:6], key=lambda item: item[1])
    context = "\n\n".join(item[2] for item in selected)

    system_prompt = """
You are PDF Study AI, a clear and accurate academic tutor.

Use the uploaded PDF excerpts as your primary source.

For questions about a topic, structure your response using:
1. Complete definition
2. Detailed explanation in simple English
3. Main characteristics or key points
4. Types, steps, or examples where relevant
5. Advantages and disadvantages where relevant
6. A comparison table when a meaningful comparison exists
7. A short exam-ready answer

For comparison questions, compare the requested concepts directly.
Do not force comparisons when they are not useful.

If the supplied excerpts do not contain the answer, say so clearly.
Do not pretend you have searched the entire PDF if the selected excerpts
do not support the answer. Label any extra general knowledge separately.

Never invent quotations or page numbers. Correct misunderstandings
politely. Use Markdown headings, bullet lists, and tables.
"""

    prompt = f"""
STUDENT QUESTION:
{question}

RELEVANT EXCERPTS FROM THE UPLOADED PDF:
{context}

Answer in simple, clear, exam-friendly English.
"""

    return call_groq(
        client,
        system_prompt,
        prompt,
        history=history
    )


# =========================================================
# EXAM PRACTICE
# =========================================================

def generate_practice(client, chunks, count, difficulty, question_type):
    # Sample from across the document, rather than only its beginning.
    if len(chunks) <= 8:
        selected = chunks
    else:
        indexes = [
            round(i * (len(chunks) - 1) / 7)
            for i in range(8)
        ]
        selected = [chunks[i] for i in indexes]

    material = "\n\n".join(selected)

    prompt = f"""
Create {count} {difficulty.lower()} exam-practice questions.

Question format: {question_type}

Use this course material:
{material[:45000]}

Requirements:
- Cover different topics from the provided material.
- Use clear student-friendly language.
- Number every question.
- For MCQs, provide four options labelled A, B, C, D.
- For short-answer questions, test understanding of key concepts.
- For long-answer questions, ask for definitions, explanations,
  steps, or comparisons.
- Put the answer key under a separate heading at the end.
- Explain answers where useful.
- Do not invent course-specific facts.
"""

    return call_groq(
        client,
        "You create accurate exam practice from supplied study notes.",
        prompt
    )


# =========================================================
# SESSION STATE
# =========================================================

for key, default in {
    "messages": [],
    "pdf_signature": None,
    "pdf_text": "",
    "pdf_chunks": [],
    "page_count": 0,
    "ocr_pages": 0,
    "topic_outline": None,
    "practice_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("📚 Study Dashboard")
    st.caption("Your PDF-powered revision assistant")

    uploaded_pdf = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"],
        help="Upload a text-based or scanned PDF."
    )

    st.caption(
        "Large PDFs take longer to process. Gemini API usage limits "
        "and charges may apply."
    )

    st.divider()
    st.markdown("### Features")
    st.markdown("""
    - Automatic scanned-page OCR
    - Topic extraction
    - Definitions and explanations
    - Concept comparisons
    - Exam-style practice
    - Revision chat
    """)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>PDF Study AI</h1>
    <p>Understand every topic. Compare concepts. Practise for your exam.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# CHECK UPLOAD AND KEYS
# =========================================================

if uploaded_pdf is None:
    st.info("Upload your course PDF in the sidebar to begin.")
    st.markdown("""
    ### How to use
    1. Upload your PDF notes.
    2. Let the app extract text and recognize scanned pages.
    3. Review the topic outline.
    4. Ask questions about any topic.
    5. Generate exam questions and review the answer key.
    """)
    st.stop()

groq_key = get_secret("GROQ_API_KEY")
gemini_key = get_secret("GEMINI_API_KEY")

if not groq_key:
    st.error("GROQ_API_KEY is missing from Streamlit Secrets.")
    st.stop()

if not gemini_key:
    st.error("GEMINI_API_KEY is missing from Streamlit Secrets.")
    st.stop()

groq_client = Groq(api_key=groq_key)
gemini_client = genai.Client(api_key=gemini_key)

# Hash file contents so a new PDF triggers a new extraction.
pdf_hash = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()

if st.session_state.pdf_signature != pdf_hash:
    st.session_state.pdf_signature = None
    st.session_state.pdf_text = ""
    st.session_state.pdf_chunks = []
    st.session_state.topic_outline = None
    st.session_state.practice_result = None
    st.session_state.messages = []

    try:
        with st.spinner(
            "Extracting PDF text and using Gemini OCR on scanned pages..."
        ):
            text, page_count, ocr_pages = extract_pdf_text(
                uploaded_pdf,
                gemini_client
            )

            chunks = split_text(text)

            st.session_state.pdf_text = text
            st.session_state.pdf_chunks = chunks
            st.session_state.page_count = page_count
            st.session_state.ocr_pages = ocr_pages
            st.session_state.pdf_signature = pdf_hash

    except Exception as exc:
        st.error(
            "PDF processing failed. Check your Gemini API key, model "
            "access, PDF size, and API usage limits."
        )
        st.caption(f"Technical detail: {str(exc)[:500]}")
        st.stop()

# =========================================================
# PDF SUMMARY METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric("PDF pages", st.session_state.page_count)
col2.metric("Scanned pages processed", st.session_state.ocr_pages)
col3.metric("Text chunks", len(st.session_state.pdf_chunks))

st.success(
    "PDF text extraction is complete. You can now ask questions."
)

chunks = st.session_state.pdf_chunks
pdf_text = st.session_state.pdf_text

# =========================================================
# MAIN TABS
# =========================================================

tab_chat, tab_topics, tab_practice, tab_pdf = st.tabs([
    "💬 Ask Questions",
    "📖 All Topics",
    "📝 Exam Practice",
    "📄 PDF Text"
])

# =========================================================
# CHAT TAB
# =========================================================

with tab_chat:
    st.subheader("Ask your PDF")

    quick_prompts = [
        "Explain the important topics",
        "Give a complete definition and explanation",
        "Compare related concepts",
        "Give me short exam notes",
    ]

    prompt_cols = st.columns(2)

    for index, quick_prompt in enumerate(quick_prompts):
        if prompt_cols[index % 2].button(
            quick_prompt,
            key=f"quick_{index}",
            use_container_width=True
        ):
            st.session_state.pending_question = quick_prompt

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask anything about your uploaded PDF..."
    )

    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding relevant PDF content..."):
                try:
                    history = st.session_state.messages[:-1]

                    answer = answer_question(
                        groq_client,
                        question,
                        chunks,
                        history=history
                    )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as exc:
                    st.error(
                        "Could not generate an answer. Check Groq model "
                        "access, API key, and usage limits."
                    )
                    st.caption(f"Technical detail: {str(exc)[:400]}")
                    st.session_state.messages.pop()

# =========================================================
# TOPICS TAB
# =========================================================

with tab_topics:
    st.subheader("Complete Topic Outline")

    st.write(
        "Generate an organized outline of topics and subtopics "
        "from your extracted PDF content."
    )

    if st.button("Extract All Topics", type="primary"):
        with st.spinner("Organizing topics from your PDF..."):
            try:
                st.session_state.topic_outline = build_topic_list(
                    groq_client,
                    chunks
                )
            except Exception as exc:
                st.error(
                    "Topic extraction failed. Check your Groq API key, "
                    "model availability, and usage limits."
                )
                st.caption(f"Technical detail: {str(exc)[:400]}")

    if st.session_state.topic_outline:
        st.markdown(st.session_state.topic_outline)

        st.download_button(
            "Download topic outline",
            data=st.session_state.topic_outline,
            file_name="pdf_topic_outline.md",
            mime="text/markdown"
        )

# =========================================================
# PRACTICE TAB
# =========================================================

with tab_practice:
    st.subheader("Exam Practice")
    st.write(
        "Generate practice questions from the extracted study material."
    )

    question_type = st.selectbox(
        "Question format",
        [
            "Mixed questions",
            "Multiple-choice questions (MCQs)",
            "Short-answer questions",
            "Long-answer questions"
        ]
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard"]
    )

    question_count = st.slider(
        "Number of questions",
        min_value=5,
        max_value=20,
        value=10
    )

    if st.button("Generate Practice Questions", type="primary"):
        with st.spinner("Creating practice questions..."):
            try:
                st.session_state.practice_result = generate_practice(
                    groq_client,
                    chunks,
                    question_count,
                    difficulty,
                    question_type
                )
            except Exception as exc:
                st.error(
                    "Practice generation failed. Check your Groq API "
                    "key, model access, and usage limits."
                )
                st.caption(f"Technical detail: {str(exc)[:400]}")

    if st.session_state.practice_result:
        st.markdown(st.session_state.practice_result)

        st.download_button(
            "Download practice questions",
            data=st.session_state.practice_result,
            file_name="exam_practice.md",
            mime="text/markdown"
        )

# =========================================================
# EXTRACTED PDF TEXT TAB
# =========================================================

with tab_pdf:
    st.subheader("Extracted PDF Text")

    st.caption(
        "Previewing the first 15,000 characters of the extracted text."
    )

    st.text_area(
        "Recognized text",
        value=pdf_text[:15000],
        height=450
    )

    st.download_button(
        "Download all extracted text",
        data=pdf_text,
        file_name="extracted_pdf_text.txt",
        mime="text/plain"
    )

st.divider()

st.caption(
    "PDF Study AI | AI-generated answers can contain errors. "
    "Verify important exam details against your original notes."
)
