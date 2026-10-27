
import os
import re
import streamlit as st
import fitz
from groq import Groq

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="PDF Study AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM DESIGN --------------------
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
    background: rgba(255,255,255,0.8);
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

# -------------------- CONFIGURATION --------------------
MODEL_NAME = "llama-3.3-70b-versatile"
MAX_PDF_CHARS = 100_000
MAX_HISTORY_MESSAGES = 8


def get_api_key():
    """Read the API key from Streamlit Secrets or environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "")


def extract_pdf_text(uploaded_file):
    """Extract text from all readable pages of the uploaded PDF."""
    text_parts = []
    page_count = 0

    try:
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as pdf:
            page_count = len(pdf)

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()

                if page_text:
                    text_parts.append(
                        f"\n--- PAGE {page_number} ---\n{page_text}"
                    )

    except Exception as exc:
        raise ValueError(
            "The PDF could not be opened. Please upload a valid PDF."
        ) from exc

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        raise ValueError(
            "No readable text was found. If your PDF contains scanned "
            "images, OCR is needed before this app can read it."
        )

    return full_text, page_count


def split_text(text, chunk_size=11000, overlap=800):
    """Split extracted text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Try to end near a paragraph boundary.
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


def call_groq(client, system_prompt, user_prompt, history=None):
    """Send a request to the Groq chat completion API."""
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=3500
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The AI returned an empty response.")

    return answer


def build_topic_list(client, chunks):
    """Ask the model to identify topics from every PDF section."""
    all_topics = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""
Read this section of the student's PDF.

SECTION {index} OF {len(chunks)}:
{chunk}

Identify all distinct:
- Main topics and subtopics
- Definitions and key terms
- Models, theories, and frameworks
- Processes, formulas, and classifications
- Important examples and comparisons

Return a concise bullet list of topic names only.
Include only topics supported by this section.
Do not invent topics.
"""
        result = call_groq(
            client,
            "You are a careful academic PDF topic extractor.",
            prompt
        )
        all_topics.append(f"SECTION {index}\n{result}")

    combined = "\n\n".join(all_topics)

    # Ask the model to merge duplicates across sections.
    final_topics = call_groq(
        client,
        "You organize study topics accurately and avoid inventing information.",
        f"""
Merge the following extracted topic lists into one organized outline.
Remove duplicates, preserve important subtopics, and group related
topics together. Use numbered headings and bullet points.

EXTRACTED LISTS:
{combined[:50000]}

Return the outline only.
"""
    )

    return final_topics


def answer_question(client, question, chunks, history=None):
    """
    Find relevant PDF sections using keyword matching, then ask Groq
    to answer using those sections.
    """
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

    # Include the best matching chunks; retain PDF order for readability.
    selected = sorted(ranked[:5], key=lambda item: item[1])
    context = "\n\n".join(item[2] for item in selected)

    system_prompt = """
You are PDF Study AI, a patient and accurate academic tutor.

IMPORTANT RULES:
1. Use the supplied PDF excerpts as the primary source.
2. Never claim you have covered every detail if the supplied excerpts
   do not support that claim.
3. If the answer is not available in the excerpts, say so clearly.
   You may give general background, but label it separately.
4. Explain in simple English suitable for a student preparing for an exam.
5. For a topic question, organize the response using:
   - Definition
   - Detailed explanation
   - Key points or characteristics
   - Types, steps, or examples, when relevant
   - Advantages and disadvantages, when relevant
   - Comparison table, when a meaningful comparison exists
   - Short exam-ready answer
6. Do not force a comparison when no meaningful comparison exists.
7. For a comparison request, compare the concepts directly in a table.
8. For a practice request, ask questions based on the PDF.
9. Correct misunderstandings politely and explain the correction.
10. Never invent quotations, page numbers, or facts from the PDF.
11. Use Markdown headings, lists, and tables for readability.
12. Cover all relevant details found in the supplied excerpts.
"""

    prompt = f"""
STUDENT'S QUESTION:
{question}

RELEVANT EXTRACTS FROM THE UPLOADED PDF:
{context}

Answer the question using the rules in your system instructions.
"""

    return call_groq(client, system_prompt, prompt, history)


def generate_practice(client, chunks, count, difficulty, practice_type):
    """Create exam practice using sections from across the PDF."""
    sample = "\n\n".join(
        chunks[:min(len(chunks), 8)]
    )

    return call_groq(
        client,
        "You are an academic exam-practice creator. Use only the provided material.",
        f"""
Create {count} {difficulty.lower()} practice questions for exam revision.

Question type: {practice_type}

Use this PDF material:
{sample[:45000]}

Requirements:
- Cover different topics, not just one section.
- Make questions clear and suitable for a student.
- Number every question.
- For multiple-choice questions, give four options labelled A, B, C, D.
- For short-answer questions, ask questions that test understanding.
- For long questions, ask for definitions, explanations, or comparisons.
- Provide a separate ANSWER KEY at the end.
- Explain why answers are correct where useful.
- Do not invent information absent from the PDF.
"""
    )


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("📚 Study Dashboard")
    st.caption("Your PDF-powered revision assistant")

    uploaded_pdf = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"],
        help="Upload a text-based PDF containing your course notes."
    )

    st.divider()
    st.markdown("### Features")
    st.markdown("""
    - Topic extraction
    - Definitions and explanations
    - Concept comparisons
    - Exam-style practice
    - Revision chat
    """)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -------------------- MAIN HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>PDF Study AI</h1>
    <p>Understand every topic. Compare concepts. Practise for your exam.</p>
</div>
""", unsafe_allow_html=True)

if not uploaded_pdf:
    st.info("Upload your course PDF in the sidebar to begin.")
    st.markdown("### How to use")
    st.markdown("""
    1. Upload your PDF notes.
    2. Extract and review the topics.
    3. Ask questions about any topic.
    4. Study definitions, detailed explanations, and comparisons.
    5. Generate practice questions and check the answer key.
    """)
    st.stop()


# -------------------- API CLIENT --------------------
api_key = get_api_key()

if not api_key:
    st.error(
        "Groq API key not found. Add GROQ_API_KEY to Streamlit Secrets "
        "or your environment variables."
    )
    st.stop()

client = Groq(api_key=api_key)

# -------------------- READ PDF --------------------
pdf_signature = (
    uploaded_pdf.name,
    uploaded_pdf.size,
    uploaded_pdf.getvalue()[:100]
)

if st.session_state.get("pdf_signature") != pdf_signature:
    with st.spinner("Reading your PDF..."):
        try:
            pdf_text, page_count = extract_pdf_text(uploaded_pdf)

            st.session_state.pdf_text = pdf_text
            st.session_state.pdf_chunks = split_text(pdf_text)
            st.session_state.pdf_signature = pdf_signature
            st.session_state.page_count = page_count
            st.session_state.topic_outline = None
            st.session_state.messages = []
            st.session_state.practice_result = None

        except ValueError as exc:
            st.error(str(exc))
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "practice_result" not in st.session_state:
    st.session_state.practice_result = None

pdf_text = st.session_state.pdf_text
chunks = st.session_state.pdf_chunks

# -------------------- PDF INFORMATION --------------------
col1, col2, col3 = st.columns(3)

col1.metric("PDF pages", st.session_state.page_count)
col2.metric("Text chunks", len(chunks))
col3.metric("Extracted characters", f"{len(pdf_text):,}")

if len(pdf_text) > MAX_PDF_CHARS:
    st.warning(
        "This is a large PDF. The app can read it, but the topic extractor "
        "may require several API requests and can reach usage limits."
    )

tab_chat, tab_topics, tab_practice, tab_pdf = st.tabs([
    "💬 Ask Questions",
    "📖 All Topics",
    "📝 Exam Practice",
    "📄 PDF Preview"
])

# -------------------- CHAT TAB --------------------
with tab_chat:
    st.subheader("Ask your PDF")

    quick_prompts = [
        "Explain the most important topics",
        "Give me a complete definition and explanation",
        "Compare the important concepts",
        "Give me short exam notes"
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
            with st.spinner("Studying your PDF..."):
                try:
                    recent_history = st.session_state.messages[:-1]
                    response = answer_question(
                        client,
                        question,
                        chunks,
                        history=recent_history
                    )
                    st.markdown(response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as exc:
                    st.error(
                        "The answer could not be generated. Check your "
                        "Groq model access, API key, and usage limits."
                    )
                    st.caption(f"Technical detail: {str(exc)[:300]}")
                    st.session_state.messages.pop()

# -------------------- TOPICS TAB --------------------
with tab_topics:
    st.subheader("Complete Topic Outline")
    st.write(
        "Generate an organized outline of the main topics and subtopics "
        "found throughout your PDF."
    )

    if st.button("Extract All Topics", type="primary"):
        with st.spinner("Analysing PDF sections and organizing topics..."):
            try:
                outline = build_topic_list(client, chunks)
                st.session_state.topic_outline = outline
            except Exception as exc:
                st.error(
                    "Topic extraction failed. Check the Groq API key, "
                    "model access, and usage limits."
                )
                st.caption(f"Technical detail: {str(exc)[:300]}")

    if st.session_state.get("topic_outline"):
        st.markdown(st.session_state.topic_outline)
        st.download_button(
            "Download topic outline",
            data=st.session_state.topic_outline,
            file_name="pdf_topic_outline.md",
            mime="text/markdown"
        )

# -------------------- PRACTICE TAB --------------------
with tab_practice:
    st.subheader("Exam Practice")
    st.write("Generate questions from your PDF and review the answer key.")

    practice_type = st.selectbox(
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
        with st.spinner("Creating exam practice from your PDF..."):
            try:
                result = generate_practice(
                    client,
                    chunks,
                    question_count,
                    difficulty,
                    practice_type
                )
                st.session_state.practice_result = result
            except Exception as exc:
                st.error(
                    "Practice generation failed. Check the API key, "
                    "model access, and usage limits."
                )
                st.caption(f"Technical detail: {str(exc)[:300]}")

    if st.session_state.practice_result:
        st.markdown(st.session_state.practice_result)
        st.download_button(
            "Download practice questions",
            data=st.session_state.practice_result,
            file_name="exam_practice.md",
            mime="text/markdown"
        )

# -------------------- PDF TAB --------------------
with tab_pdf:
    st.subheader("PDF Content Preview")
    st.caption(
        "Showing extracted text. This preview is limited to the first "
        "12,000 characters."
    )

    st.text_area(
        "Extracted PDF text",
        value=pdf_text[:12000],
        height=450
    )

    st.download_button(
        "Download extracted text",
        data=pdf_text,
        file_name="extracted_pdf_text.txt",
        mime="text/plain"
    )

st.divider()
st.caption(
    "PDF Study AI | Answers are generated by AI. Verify important "
    "exam details against your original course notes."
)
