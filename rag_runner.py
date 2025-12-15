#!/usr/bin/env python3
# regulatory_update_tracker.py
# AI-Powered Regulatory Compliance Checker — RAG with Groq (LLAMA 3.3)
# Updated with GDPR & HIPAA compliance awareness

import os
import re
import json
import textwrap
from pathlib import Path
from dotenv import load_dotenv
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.colors import red, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
load_dotenv()
DOCS_PATH     = Path("./contracts_Dataset")
REGS_PATH     = Path("./regulations_Dataset")  # optional, used only if available
INDEX_PATH    = Path("./faiss_index")
REBUILD_INDEX = True
MODEL_NAME    = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 4
WRAP_WIDTH    = 95  # approximate characters per line for PDF wrapping

# Ensure fonts for bold supported
try:
    pdfmetrics.registerFont(TTFont("Helvetica", "Helvetica.ttf"))
    pdfmetrics.registerFont(TTFont("Helvetica-Bold", "Helvetica-Bold.ttf"))
except Exception:
    pass

# ─────────────────────────────────────────────
# UTILS: LOAD DOCUMENT(S)
# ─────────────────────────────────────────────
def load_all_pdfs(path: Path):
    docs = []

    if not path.exists():
        raise FileNotFoundError(f"❌ Path does not exist: {path}")

    if path.is_file() and path.suffix.lower() == ".pdf":
        print(f"📄 Loading single PDF: {path.name}")
        return PyPDFLoader(str(path)).load()

    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.suffix.lower() == ".pdf":
                try:
                    print(f"📄 Loading PDF: {p.name}")
                    docs.extend(PyPDFLoader(str(p)).load())
                except Exception as e:
                    print(f"[WARN] Failed to load {p.name}: {e}")
            elif p.suffix.lower() in [".txt", ".md"]:
                try:
                    print(f"📄 Loading text file: {p.name}")
                    docs.extend(TextLoader(str(p), encoding="utf-8").load())
                except Exception as e:
                    print(f"[WARN] Failed to load {p.name}: {e}")
        return docs

    raise ValueError(f"❌ Unsupported path type: {path}")


def load_one_doc(path: Path):
    docs = load_all_pdfs(path)
    if not docs:
        raise FileNotFoundError(f"❌ No valid pages extracted from: {path}")
    return docs


def extract_text(docs):
    return "\n\n".join([d.page_content for d in docs])

# ─────────────────────────────────────────────
# SPLIT + VECTORSTORE (RAG)
# ─────────────────────────────────────────────
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def build_or_load_faiss(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if REBUILD_INDEX or not INDEX_PATH.exists():
        print("🔁 Building FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(INDEX_PATH))
    else:
        print("📦 Loading FAISS index...")
        vectorstore = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# ─────────────────────────────────────────────
# HELPER: CLEAN JSON-LIKE LLM OUTPUT
# ─────────────────────────────────────────────
def extract_json_from_text(text):
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9]*\n?", "", candidate)
    if candidate.endswith("```"):
        candidate = re.sub(r"\n?```$", "", candidate)

    for start_char in ("[", "{"):
        start_idx = candidate.find(start_char)
        if start_idx != -1:
            stack = []
            for i in range(start_idx, len(candidate)):
                ch = candidate[i]
                if ch in "[{":
                    stack.append(ch)
                elif ch in "]}":
                    if not stack:
                        break
                    stack.pop()
                if not stack:
                    maybe = candidate[start_idx:i+1]
                    try:
                        return json.loads(maybe)
                    except Exception:
                        break
    try:
        return json.loads(candidate)
    except Exception:
        return None

# ─────────────────────────────────────────────
# AMENDMENT GENERATION (GDPR / HIPAA aware)
# ─────────────────────────────────────────────
def generate_amendments(llm, full_text, jurisdiction="global", laws=None):
    laws = laws or []
    laws_text = ""
    if 'GDPR' in laws:
        laws_text += "- GDPR: EU personal data privacy.\n"
    if 'HIPAA' in laws:
        laws_text += "- HIPAA: US protected health information.\n"

    prompt = f"""
You are a compliance checker. Jurisdiction: {jurisdiction}

Focus on the following laws when checking the contract:
{laws_text if laws_text else '- General compliance'}

Identify risky clauses in the contract text below. For each risky clause provide:
- clause_id: short identifier (heading or clause number if available)
- old_clause: exact text from the contract
- new_clause: compliant replacement text or empty if removing
- action: "replace" or "remove"

Return ONLY a valid JSON array.

Contract Text:
{full_text}
"""
    print("⏳ Asking LLM for amendments (including GDPR/HIPAA if selected)...")
    resp = llm.invoke(prompt)
    content = resp.content.strip()
    amendments = extract_json_from_text(content)
    if amendments is None:
        followup = "Return ONLY JSON array as described, no extra text."
        resp2 = llm.invoke(followup + "\n\nPrevious response:\n" + content)
        amendments = extract_json_from_text(resp2.content.strip())

    if amendments is None:
        print("[ERROR] Could not parse JSON from LLM response. Returning empty amendment list.")
        return []

    validated = []
    for item in amendments:
        if not isinstance(item, dict):
            continue
        clause_id = item.get("clause_id") or item.get("id") or "unknown"
        old_clause = item.get("old_clause", "").strip()
        new_clause = item.get("new_clause", "").strip() if item.get("new_clause") else ""
        action = item.get("action", "").lower()
        if action not in ("replace", "remove"):
            action = "replace" if new_clause else "remove"
        validated.append({
            "clause_id": clause_id,
            "old_clause": old_clause,
            "new_clause": new_clause,
            "action": action
        })
    return validated

# ─────────────────────────────────────────────
# REPLACE / REMOVE CLAUSES
# ─────────────────────────────────────────────
def replace_or_remove_clauses(full_text, amendments):
    corrected = full_text
    log = []
    for item in amendments:
        old = item.get("old_clause", "").strip()
        new = item.get("new_clause", "").strip()
        action = item.get("action", "replace").lower()
        clause_id = item.get("clause_id", "unknown")
        if not old:
            log.append({"clause_id": clause_id, "status": "skipped_no_old_clause"})
            continue
        pattern_exact = re.escape(old)
        matches_exact = len(re.findall(pattern_exact, corrected, flags=re.MULTILINE))
        if matches_exact > 0:
            if action == "remove":
                corrected = re.sub(pattern_exact, "", corrected)
                log.append({"clause_id": clause_id, "status": "removed", "occurrences": matches_exact})
            else:
                replacement = f"<<HIGHLIGHT_START>>{new}<<HIGHLIGHT_END>>"
                corrected = re.sub(pattern_exact, replacement, corrected)
                log.append({"clause_id": clause_id, "status": "replaced", "occurrences": matches_exact})
            continue
        # Fallback: whitespace-normalized match
        def collapse_ws(s): return re.sub(r"\s+", " ", s).strip()
        collapsed_corrected = collapse_ws(corrected)
        collapsed_old = collapse_ws(old)
        if collapsed_old in collapsed_corrected:
            occurrences = 0
            idx = 0
            while True:
                pos = collapsed_corrected.find(collapsed_old, idx)
                if pos == -1:
                    break
                sample = " ".join(collapsed_old.split()[:6])
                sample_pos = corrected.find(sample)
                if sample_pos == -1:
                    sample_pos = corrected.find(collapsed_old.split()[0])
                    if sample_pos == -1:
                        break
                window_start = max(0, sample_pos - 50)
                window_end = min(len(corrected), sample_pos + len(collapsed_old) + 200)
                window = corrected[window_start:window_end]
                found_span = None
                for a in range(0, len(window)//2):
                    for b in range(len(window)//2, len(window)):
                        candidate = window[a:b]
                        if collapse_ws(candidate) == collapsed_old:
                            if action == "remove":
                                corrected = corrected.replace(candidate, "")
                                log.append({"clause_id": clause_id, "status": "removed", "occurrences": 1})
                            else:
                                replacement = f"<<HIGHLIGHT_START>>{new}<<HIGHLIGHT_END>>"
                                corrected = corrected.replace(candidate, replacement)
                                log.append({"clause_id": clause_id, "status": "replaced", "occurrences": 1})
                            occurrences += 1
                            break
                    if occurrences:
                        break
                if occurrences == 0:
                    break
                collapsed_corrected = collapse_ws(corrected)
                idx = pos + 1
            if occurrences == 0:
                log.append({"clause_id": clause_id, "status": "not_found_after_ws_attempt"})
            continue
        log.append({"clause_id": clause_id, "status": "not_found"})
    return corrected, log

# ─────────────────────────────────────────────
# SAVE PDF WITH HIGHLIGHTS
# ─────────────────────────────────────────────
def save_pdf_with_highlights(text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=LETTER)
    width, height = LETTER
    left_margin = 50
    top_margin = height - 50
    line_height = 14
    current_y = top_margin

    def draw_wrapped(seg_text, color=black):
        nonlocal current_y
        wrapper = textwrap.TextWrapper(width=WRAP_WIDTH, replace_whitespace=False)
        lines = wrapper.wrap(seg_text)
        for ln in lines:
            if current_y < 50:
                c.showPage()
                current_y = top_margin
           
            else:
                try:
                    c.setFont("Helvetica", 10)
                except Exception:
                    c.setFont("Times-Roman", 10)
            c.setFillColor(color)
            c.drawString(left_margin, current_y, ln)
            current_y -= line_height

    paragraphs = text.split("\n")
    for para in paragraphs:
        segments = re.split(r"(<<HIGHLIGHT_START>>.*?<<HIGHLIGHT_END>>)", para)
        for seg in segments:
            if not seg:
                continue
            if seg.startswith("<<HIGHLIGHT_START>>") and seg.endswith("<<HIGHLIGHT_END>>"):
                clause_text = seg.replace("<<HIGHLIGHT_START>>", "").replace("<<HIGHLIGHT_END>>", "")
                draw_wrapped(clause_text, color=red)
            else:
                draw_wrapped(seg, color=black)
        current_y -= line_height / 2
    c.save()

# ─────────────────────────────────────────────
# RAG + QA
# ─────────────────────────────────────────────
def retrieve_context_and_answer(llm, vectorstore, question, k=TOP_K):
    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception:
        try:
            docs = vectorstore.similarity_search_with_score(question, k=k)
            docs = [d for d, _ in docs]
        except Exception:
            docs = []

    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs]) if docs else ""
    prompt = f"""You are a compliance analysis assistant.
Use the retrieved context and answer the question precisely. If unknown, say "I don't know."

Question:
{question}

Context:
{context_text}
"""
    resp = llm.invoke(prompt)
    return resp.content

# ─────────────────────────────────────────────
# CLAUSE RISK SCORING
# ─────────────────────────────────────────────
def compute_clause_risk_scores(contract_text: str, llm: ChatGroq = None):
    clause_texts = re.split(r'\n\d{1,3}\.|\n[A-Z]{2,}[:]', contract_text)
    scores = []
    if llm:
        for idx, clause in enumerate(clause_texts, start=1):
            if not clause.strip():
                continue
            prompt = f"Evaluate compliance risk 0-10 for this clause. Respond only with number:\n{clause}"
            resp = llm.invoke(prompt)
            try:
                score = float(re.findall(r"\d+\.?\d*", resp.content)[0])
            except:
                score = 5.0
            scores.append({'clause_id': f"Clause {idx}", 'risk_score': score})
    else:
        risk_keywords = ['penalty', 'liability', 'termination', 'breach', 'indemnify', 'arbitration']
        for idx, clause in enumerate(clause_texts, start=1):
            count = sum(clause.lower().count(word) for word in risk_keywords)
            score = min(count * 2 + 1, 10)
            scores.append({'clause_id': f"Clause {idx}", 'risk_score': float(score)})
    return scores

# ─────────────────────────────────────────────
# CLI MENU
# ─────────────────────────────────────────────
def cli_menu():
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("❌ Please set GROQ_API_KEY in .env")

    docs = load_one_doc(DOCS_PATH)
    full_text = extract_text(docs)
    contract_name = Path(docs[0].metadata.get("source", "contract")).stem

    regs_docs = load_all_pdfs(REGS_PATH) if REGS_PATH.exists() else []
    regs_text = extract_text(regs_docs) if regs_docs else ""

    try:
        chunks = split_docs(docs + regs_docs) if regs_docs else split_docs(docs)
        vectorstore = build_or_load_faiss(chunks)
    except Exception as e:
        print(f"[WARN] FAISS build/load failed: {e}")
        vectorstore = None

    llm = ChatGroq(model=MODEL_NAME, temperature=0.2)
    current_text = full_text
    last_log = []

    while True:
        print("\n=== COMPLIANCE CHECKER MENU ===")
        print("1) Quick analysis (LLM summary & issues)")
        print("2) Generate amendments (jurisdiction + laws)")
        print("3) Save corrected PDF (bold red highlights)")
        print("4) Compliance Q&A (RAG + LLM)")
        print("5) Show last replacement log")
        print("6) Exit")
        choice = input("👉 Select option: ").strip()

        if choice == "1":
            print("⏳ Running quick analysis...")
            resp = llm.invoke(f"Summarize and list compliance issues from the contract:\n\n{current_text}")
            print("\n🔍 Analysis:\n", resp.content)

        elif choice == "2":
            jurisdiction = input("🌍 Enter jurisdiction [default: global]: ").strip() or "global"
            laws_input = input("⚖️ Specify laws (comma-separated, e.g., GDPR,HIPAA) [optional]: ").strip()
            laws_list = [law.strip() for law in laws_input.split(",")] if laws_input else []
            amendments = generate_amendments(llm, current_text, jurisdiction=jurisdiction, laws=laws_list)
            if not amendments:
                print("✅ No risky clauses found or LLM returned none.")
                last_log = []
            else:
                corrected, log = replace_or_remove_clauses(current_text, amendments)
                current_text = corrected
                last_log = log
                print("✅ Amendments applied. Log:")
                for item in log:
                    print(" -", item)

        elif choice == "3":
            output_path = DOCS_PATH / f"{contract_name}_AMENDED.pdf"
            save_pdf_with_highlights(current_text, output_path)
            print(f"✅ Corrected PDF saved at: {output_path}")

        elif choice == "4":
            if vectorstore is None:
                print("[WARN] Vectorstore unavailable. Using full-text LLM.")
                while True:
                    q = input("❓ Ask about compliance (type 'exit' to quit): ")
                    if q.lower() in ("exit", "quit"):
                        break
                    resp = llm.invoke(f"Answer using the contract text. If unknown, say 'I don't know'.\n\nContract:\n{current_text}\n\nQuestion:\n{q}")
                    print("\n🧠 Answer:\n", resp.content)
            else:
                while True:
                    q = input("❓ Ask about compliance (type 'exit' to quit): ")
                    if q.lower() in ("exit", "quit"):
                        break
                    ans = retrieve_context_and_answer(llm, vectorstore, q)
                    print("\n🧠 Answer:\n", ans)

        elif choice == "5":
            if last_log:
                for item in last_log:
                    print(item)
            else:
                print("⚠️ No replacement log yet.")

        elif choice == "6":
            print("👋 Exiting...")
            break

        else:
            print("❌ Invalid choice.")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cli_menu()
