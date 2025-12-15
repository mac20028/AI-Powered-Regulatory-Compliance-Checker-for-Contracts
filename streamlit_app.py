# streamlit_app.py
import streamlit as st
from pathlib import Path
import tempfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from rag_runner import (
    load_one_doc, extract_text, split_docs, build_or_load_faiss,
    generate_amendments, replace_or_remove_clauses, save_pdf_with_highlights,
    compute_clause_risk_scores, retrieve_context_and_answer, MODEL_NAME
)
from email_notify import notify_compliance_update

from langchain_groq import ChatGroq
import base64

# ---------------------------
# PAGE CONFIG & CSS
# ---------------------------
st.set_page_config(
    page_title="AI Compliance Checker", 
    layout="wide", 
    page_icon="⚖",
    initial_sidebar_state="expanded"
)

# Clean Professional Theme CSS
st.markdown("""
<style>
/* Color Variables */
:root {
    --primary-blue: #1a237e;
    --primary-blue-light: #283593;
    --secondary-green: #2e7d32;
    --secondary-green-light: #4caf50;
    --accent-purple: #4527a0;
    --warning-orange: #f57c00;
    --warning-orange-light: #ff9800;
    --gray-light: #f5f5f5;
    --gray-medium: #e0e0e0;
    --gray-dark: #424242;
    --white: #ffffff;
}

/* Global Styles */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}

/* Header */
.header {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--primary-blue);
    margin-bottom: 0.5rem;
    border-bottom: 2px solid var(--gray-medium);
    padding-bottom: 0.5rem;
}

.subtitle {
    font-size: 1rem;
    color: var(--gray-dark);
    margin-bottom: 1.5rem;
}

/* Card Design */
.card {
    background: var(--white);
    padding: 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid var(--gray-medium);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--primary-blue);
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--gray-medium);
}

/* Button Styles */
.btn-primary {
    background-color: var(--primary-blue);
    color: var(--white);
    font-weight: 500;
    padding: 0.6rem 1.2rem;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    transition: background-color 0.2s;
    font-size: 0.95rem;
}

.btn-primary:hover {
    background-color: var(--primary-blue-light);
}

.btn-secondary {
    background-color: var(--secondary-green);
    color: var(--white);
    font-weight: 500;
    padding: 0.6rem 1.2rem;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    transition: background-color 0.2s;
    font-size: 0.95rem;
}

.btn-secondary:hover {
    background-color: var(--secondary-green-light);
}

.btn-warning {
    background-color: var(--warning-orange);
    color: var(--white);
    font-weight: 500;
    padding: 0.6rem 1.2rem;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    transition: background-color 0.2s;
    font-size: 0.95rem;
}

.btn-warning:hover {
    background-color: var(--warning-orange-light);
}

/* Clause Display */
.old-clause {
    background: rgba(245, 124, 0, 0.05);
    padding: 1rem;
    border-left: 3px solid var(--warning-orange);
    border-radius: 4px;
    margin-bottom: 0.5rem;
    font-family: monospace;
    font-size: 0.9rem;
    line-height: 1.4;
}

.new-clause {
    background: rgba(46, 125, 50, 0.05);
    padding: 1rem;
    border-left: 3px solid var(--secondary-green);
    border-radius: 4px;
    margin-bottom: 0.5rem;
    font-family: monospace;
    font-size: 0.9rem;
    line-height: 1.4;
}

/* Amendment Card Styles */
.amendment-card {
    background: var(--white);
    padding: 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid var(--gray-medium);
}

.amendment-card .old {
    background: rgba(245, 124, 0, 0.05);
    padding: 0.8rem;
    border-left: 3px solid var(--warning-orange);
    border-radius: 4px;
    margin: 0.5rem 0;
    font-family: monospace;
    font-size: 0.9rem;
    line-height: 1.4;
}

.amendment-card .new {
    background: rgba(46, 125, 50, 0.05);
    padding: 0.8rem;
    border-left: 3px solid var(--secondary-green);
    border-radius: 4px;
    margin: 0.5rem 0;
    font-family: monospace;
    font-size: 0.9rem;
    line-height: 1.4;
}

/* PDF Preview - Large */
.pdf-preview-large {
    width: 100%;
    height: 700px;
    border: 1px solid var(--gray-medium);
    border-radius: 4px;
    margin-top: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Navigation */
.sidebar-nav {
    padding: 0.5rem 0;
}

.nav-item {
    padding: 0.6rem 1rem;
    margin: 0.2rem 0;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--gray-dark);
    font-weight: 500;
}

.nav-item:hover {
    background-color: rgba(26, 35, 126, 0.05);
    color: var(--primary-blue);
}

.nav-item.active {
    background-color: var(--primary-blue);
    color: var(--white);
}

/* Status Indicators */
.status {
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-size: 0.9rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

.status-success {
    background-color: rgba(46, 125, 50, 0.1);
    color: var(--secondary-green);
}

.status-warning {
    background-color: rgba(245, 124, 0, 0.1);
    color: var(--warning-orange);
}

.status-info {
    background-color: rgba(26, 35, 126, 0.1);
    color: var(--primary-blue);
}

/* Text Areas */
.contract-text {
    font-family: monospace;
    font-size: 0.9rem;
    line-height: 1.5;
    padding: 1rem;
    background: var(--gray-light);
    border: 1px solid var(--gray-medium);
    border-radius: 4px;
    overflow-y: auto;
}

/* Chat Messages */
.chat-user {
    background: rgba(26, 35, 126, 0.05);
    padding: 0.8rem 1rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    border-left: 3px solid var(--primary-blue);
}

.chat-bot {
    background: var(--gray-light);
    padding: 0.8rem 1rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    border-left: 3px solid var(--gray-medium);
}

/* Responsive */
@media (max-width: 768px) {
    .header {
        font-size: 1.8rem;
    }
    
    .pdf-preview-large {
        height: 500px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.markdown("""
<div style='margin-bottom: 1.2rem;'>
    <h1 style='color: #1a237e; margin-bottom: 0.1rem;font-size: 3.0rem'>AI Compliance</h1>
    <h3 style='color: #666; font-size: 0.9rem; margin: 0;'>Contract Analysis Tool</h3>
</div>
""", unsafe_allow_html=True)

# Navigation options
pages = [
    "Home",
    "Upload Contract",
    "Generate Amendments",
    "Risk Dashboard",
    "Chat with Contract",
    "Help"
]

# Initialize session state for page
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Create navigation buttons
st.sidebar.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
for page in pages:
    if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True, 
                         type="primary" if st.session_state.current_page == page else "secondary"):
        st.session_state.current_page = page
        st.rerun()
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Status")

if "contract_text" in st.session_state:
    st.sidebar.markdown("<div class='status status-success'>Contract Loaded</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div class='status status-warning'>No Contract</div>", unsafe_allow_html=True)

if "amendments" in st.session_state and st.session_state["amendments"]:
    st.sidebar.markdown(f"<div class='status status-success'>{len(st.session_state['amendments'])} Amendments</div>", unsafe_allow_html=True)

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_resource(show_spinner=False)
def init_llm():
    return ChatGroq(model=MODEL_NAME, temperature=0.2)

@st.cache_data(show_spinner=False)
def build_index_from_docs(docs):
    chunks = split_docs(docs)
    return build_or_load_faiss(chunks)

def save_bytes_to_tempfile(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return Path(tmp.name)

def render_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe class="pdf-preview-large" src="data:application/pdf;base64,{base64_pdf}"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# ---------------------------
# Home Page
# ---------------------------
if st.session_state.current_page == "Home":
    st.markdown("<div class='header'>AI Regulatory Compliance Checker</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Automated contract analysis and compliance checking</div>", unsafe_allow_html=True)
    
    # Features in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Contract Upload</div>
            <p style='color: #666; font-size: 0.9rem;'>Upload PDF or text contracts for automated analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Compliance Check</div>
            <p style='color: #666; font-size: 0.9rem;'>Check for GDPR, HIPAA and jurisdiction requirements.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>Risk Analysis</div>
            <p style='color: #666; font-size: 0.9rem;'>Visual risk scoring and improvement tracking.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use
    st.markdown("## How to Use")
    
    steps = [
        {"title": "Upload Contract", "desc": "Upload your contract in PDF or text format."},
        {"title": "Select Regulations", "desc": "Choose applicable regulations and jurisdiction."},
        {"title": "Review Amendments", "desc": "Review AI-generated compliance amendments."},
        {"title": "Export Results", "desc": "Download amended contract and risk reports."}
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"**{i}. {step['title']}**")
        st.markdown(f"<p style='color: #666; margin-left: 1rem; margin-bottom: 1rem;'>{step['desc']}</p>", unsafe_allow_html=True)
    
    # Start button
    if st.button("Start Analysis", type="primary"):
        st.session_state.current_page = "Upload Contract"
        st.rerun()

# ---------------------------
# Upload Contract Page
# ---------------------------
elif st.session_state.current_page == "Upload Contract":
    st.markdown("<div class='header'>Upload Contract</div>", unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload contract document", type=["pdf", "txt"])
    
    if uploaded:
        tmp_path = save_bytes_to_tempfile(uploaded)
        
        # Progress indicators
        with st.spinner("Loading document..."):
            docs = load_one_doc(tmp_path)
            extracted_text = extract_text(docs)
            
            # Store in session state
            st.session_state["contract_docs"] = docs
            st.session_state["contract_text"] = extracted_text
            st.session_state["uploaded_path"] = tmp_path
            
            # Build FAISS index automatically
            with st.spinner("Building search index..."):
                vs = build_index_from_docs(docs)
                st.session_state["vectorstore"] = vs
        
        st.success("Contract loaded successfully")
        
        # Contract stats
        col1, col2, col3 = st.columns(3)
        char_count = len(extracted_text)
        word_count = len(extracted_text.split())
        
        with col1:
            st.metric("Words", f"{word_count:,}")
        with col2:
            st.metric("Characters", f"{char_count:,}")
        with col3:
            st.metric("Format", uploaded.name.split('.')[-1].upper())
        
        # Contract preview
        st.markdown("### Contract Text Preview")
        st.text_area("", extracted_text[:3000], height=200, label_visibility="hidden")
        
        # PDF Preview for PDF files - Only show when button clicked
        if uploaded.name.endswith(".pdf"):
            st.markdown("### PDF Preview")
            if "show_original_pdf" not in st.session_state:
                st.session_state["show_original_pdf"] = False
            
            # Button to show/hide PDF
            if st.button("PDF Preview"):
                st.session_state["show_original_pdf"] = not st.session_state["show_original_pdf"]
                st.rerun()
            
        
          
        
        # Next step
        if st.button("Continue to Amendments", type="primary"):
            st.session_state.current_page = "Generate Amendments"
            st.rerun()
    
    else:
        st.info("Please upload a contract file to begin analysis.")

# ---------------------------
# Generate Amendments Page
# ---------------------------
elif st.session_state.current_page == "Generate Amendments":
    st.markdown("<div class='header'>Generate Amendments</div>", unsafe_allow_html=True)
    
    if "contract_text" not in st.session_state:
        st.warning("Please upload a contract first.")
        if st.button("Go to Upload"):
            st.session_state.current_page = "Upload Contract"
            st.rerun()
        st.stop()
    
    contract_text = st.session_state["contract_text"]
    llm = init_llm()
    
    # Settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card-title'>Jurisdiction</div>", unsafe_allow_html=True)
        jurisdiction = st.selectbox("", ["Global", "India", "US", "EU", "UK", "Singapore", "UAE"], label_visibility="collapsed")
    
    with col2:
        st.markdown("<div class='card-title'>Regulations</div>", unsafe_allow_html=True)
        gdpr_checked = st.checkbox("GDPR")
        hipaa_checked = st.checkbox("HIPAA")
        selected_laws = []
        if gdpr_checked: selected_laws.append("GDPR")
        if hipaa_checked: selected_laws.append("HIPAA")
    
    # Generate amendments - Changed to use your specified format
    if st.button("🧾 Generate Amendments", type="primary"):
        with st.spinner("Checking contract for required amendments..."):
            amendments = generate_amendments(llm, contract_text, jurisdiction=jurisdiction, laws=selected_laws)
            st.session_state["amendments"] = amendments
        
        if not amendments:
            st.success("✅ No amendments needed. Contract already compliant!")
        else:
            st.success(f"✅ {len(amendments)} amendments generated")
            for a in amendments:
                cid = a.get("clause_id","unknown")
                old = a.get("old_clause","")[:400].replace("\n"," ")
                new = a.get("new_clause","")[:400].replace("\n"," ")
                st.markdown(f"<div class='amendment-card'><b>{cid}</b>"
                            f"<div class='old'><b>Old:</b> {old}</div>"
                            f"<div class='new'><b>New:</b> {new}</div></div>", unsafe_allow_html=True)
    
    # Apply amendments
    if "amendments" in st.session_state and st.session_state["amendments"]:
        st.markdown("---")
        
        if st.button("Apply Amendments and Generate PDF", type="secondary"):
            with st.spinner("Generating amended PDF..."):
                corrected_text, logs = replace_or_remove_clauses(contract_text, st.session_state["amendments"])
                out_dir = Path("generated_outputs")
                out_dir.mkdir(exist_ok=True)
                pdf_path = out_dir / "amended_contract.pdf"
                
                save_pdf_with_highlights(corrected_text, pdf_path)
                
                # Store in session state
                st.session_state["corrected_text"] = corrected_text
                st.session_state["amended_pdf_path"] = str(pdf_path)
                
                st.success("Amended PDF generated successfully")
        
        # Show amended PDF - Only show when button clicked
        if "amended_pdf_path" in st.session_state:
            p = Path(st.session_state["amended_pdf_path"])
            if p.exists():
                st.markdown("### Amended PDF Preview")
                if "show_amended_pdf" not in st.session_state:
                    st.session_state["show_amended_pdf"] = False
                
                # Button to show/hide PDF
                if st.button("PDF Preview"):
                    st.session_state["show_amended_pdf"] = not st.session_state["show_amended_pdf"]
                    st.rerun()
                
                # Conditionally show PDF
                if st.session_state["show_amended_pdf"]:
                    render_pdf(p)
             
                
                # Download button
                with open(p, "rb") as f:
                    st.download_button(
                        "Download Amended PDF",
                        f,
                        file_name=p.name,
                        mime="application/pdf"
                    )
        
        # Email section
        st.markdown("---")
        st.markdown("### Send via Email")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            recipient_email = st.text_input("Recipient email", placeholder="email@company.com")
        
        with col2:
            if st.button("Send Email", type="primary"):
                if "amended_pdf_path" in st.session_state:
                    pdf_path_str = st.session_state.get("amended_pdf_path")
                    if pdf_path_str:
                        pdf_path = Path(pdf_path_str)
                        if pdf_path.exists() and recipient_email and "@" in recipient_email:
                            try:
                                pdf_bytes = notify_compliance_update(
                                    pdf_path=pdf_path,
                                    jurisdiction=jurisdiction,
                                    compliance_type=", ".join(selected_laws) if selected_laws else "General Compliance",
                                    recipient_email=recipient_email.strip()
                                )
                                st.success("Email sent successfully")
                            except Exception as e:
                                st.error(f"Failed to send email: {e}")
                        else:
                            st.error("Please enter a valid email address")
                    else:
                        st.error("Please generate the PDF first")

# ---------------------------
# Risk Dashboard Page
# ---------------------------
elif st.session_state.current_page == "Risk Dashboard":
    st.markdown("<div class='header'>Risk Dashboard</div>", unsafe_allow_html=True)
    
    if "contract_text" not in st.session_state:
        st.warning("Please upload and generate amendments first.")
        st.stop()
    
    original_text = st.session_state["contract_text"]
    corrected_text = st.session_state.get("corrected_text", original_text)
    
    # Calculate risk scores
    with st.spinner("Calculating risk scores..."):
        original_scores = compute_clause_risk_scores(original_text)
        corrected_scores = compute_clause_risk_scores(corrected_text)
    
    df_original = pd.DataFrame(original_scores)
    df_corrected = pd.DataFrame(corrected_scores)
    df_scores = pd.merge(df_original, df_corrected, on='clause_id', suffixes=('_original','_corrected'))
    
    # Risk metrics
    st.markdown("### Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        original_risk = df_scores['risk_score_original'].mean() * 10
        st.metric("Original Risk", f"{original_risk:.1f}%")
    
    with col2:
        corrected_risk = df_scores['risk_score_corrected'].mean() * 10
        st.metric("Amended Risk", f"{corrected_risk:.1f}%")
    
    with col3:
        risk_reduction = original_risk - corrected_risk
        st.metric("Risk Reduction", f"{risk_reduction:.1f}%")
    
    # Bar chart
    st.markdown("### Risk Comparison")
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        x=df_scores['clause_id'],
        y=df_scores['risk_score_original']*10,
        name='Original Risk',
        marker_color='#f57c00'
    ))
    
    fig_bar.add_trace(go.Bar(
        x=df_scores['clause_id'],
        y=df_scores['risk_score_corrected']*10,
        name='Amended Risk',
        marker_color='#2e7d32'
    ))
    
    fig_bar.update_layout(
        xaxis_title="Clause ID",
        yaxis_title="Risk Score (%)",
        barmode='group',
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gauge chart
    st.markdown("### Overall Risk Assessment")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=corrected_risk,
        delta={'reference': original_risk},
        title={'text': "Current Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2e7d32"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "red"}
            ]
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Data table
    st.markdown("### Detailed Risk Data")
    
    df_display = df_scores.copy()
    df_display['risk_score_original'] = (df_display['risk_score_original'] * 10).round(1)
    df_display['risk_score_corrected'] = (df_display['risk_score_corrected'] * 10).round(1)
    df_display['risk_reduction'] = df_display['risk_score_original'] - df_display['risk_score_corrected']
    
    st.dataframe(df_display, use_container_width=True)

# ---------------------------
# Chat with Contract Page
# ---------------------------
elif st.session_state.current_page == "Chat with Contract":
    st.markdown("<div class='header'>Chat with Contract</div>", unsafe_allow_html=True)
    
    if "contract_docs" not in st.session_state:
        st.warning("Please upload a contract first.")
        if st.button("Go to Upload"):
            st.session_state.current_page = "Upload Contract"
            st.rerun()
        st.stop()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Check if vectorstore exists
    if "vectorstore" not in st.session_state:
        with st.spinner("Building search index..."):
            st.session_state["vectorstore"] = build_index_from_docs(st.session_state["contract_docs"])
    
    vectorstore = st.session_state["vectorstore"]
    llm = init_llm()
    
    # Display chat history
    for chat in st.session_state["chat_history"]:
        st.markdown(f"<div class='chat-user'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bot'><strong>AI:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask a question about your contract...")
    
    if user_input:
        # Add user message to history
        st.session_state["chat_history"].append({"user": user_input, "bot": ""})
        
        # Get AI response
        with st.spinner("Analyzing contract..."):
            answer = retrieve_context_and_answer(llm, vectorstore, user_input)
            st.session_state["chat_history"][-1]["bot"] = answer
        
        # Rerun to display new messages
        st.rerun()
    
    # Suggested questions
    st.markdown("### Suggested Questions")
    
    questions = [
        "What are the main risks in this contract?",
        "Show data protection clauses",
        "What needs GDPR compliance?",
        "Explain liability clauses"
    ]
    
    cols = st.columns(2)
    for idx, question in enumerate(questions):
        with cols[idx % 2]:
            if st.button(question, use_container_width=True):
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                
                st.session_state["chat_history"].append({"user": question, "bot": ""})
                
                with st.spinner("Analyzing contract..."):
                    answer = retrieve_context_and_answer(llm, vectorstore, question)
                    st.session_state["chat_history"][-1]["bot"] = answer
                
                st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat History", type="secondary"):
        st.session_state["chat_history"] = []
        st.rerun()

# ---------------------------
# Help Page
# ---------------------------
else:
    st.markdown("<div class='header'>Help</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### How to Use
    
    1. **Upload Contract**
       - Go to Upload Contract page
       - Upload PDF or text file
       - System will process automatically
    
    2. **Generate Amendments**
       - Select jurisdiction and regulations
       - Click Generate Amendments
       - Review suggested changes
    
    3. **Risk Analysis**
       - View risk scores on dashboard
       - Compare original vs amended risk
       - Export risk data
    
    4. **Chat with Contract**
       - Ask questions about your contract
       - Get AI-powered answers
       - Suggested questions provided
    
    ### Supported Regulations
    
    - **GDPR**: European data protection regulation
    - **HIPAA**: US healthcare data protection
    - Jurisdiction-specific requirements
    
    ### File Requirements
    
    - PDF files with extractable text
    - Text files (.txt)
    - Maximum file size: 50MB
    
    ### Troubleshooting
    
    If you encounter issues:
    
    1. Ensure PDF has selectable text (not scanned images)
    2. Clear browser cache if app behaves unexpectedly
    3. Restart the app if unresponsive
    
    For additional help, contact support.
    """)
    
    if st.button("Return to Home", type="primary"):
        st.session_state.current_page = "Home"
        st.rerun()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.9rem;'>AI Compliance Checker | Professional Contract Analysis Tool</div>", unsafe_allow_html=True)