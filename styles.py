"""
UI Presentation, Theme & Custom Styling Module (`styles.py`).

Provides custom CSS, branding metadata, and starter prompts for the Gradio ChatInterface.
"""

import sys

sys.dont_write_bytecode = True

APP_TITLE: str = "🤖 Ajhar's AI Automation & Backend Consultancy Digital Twin"

APP_DESCRIPTION: str = (
    "Welcome! I am the production Digital Twin of **MD Ajharul Islam Akanda (Ajhar)** — "
    "**AI Automation Engineer, Full-Stack Backend Engineer & Technical Product Manager**.\n\n"
    "🛡️ **Guarded by a Multi-LLM Evaluator-Optimizer Pipeline:** Every response is audited in real time "
    "by an independent Evaluator LLM (`Pydantic v2`) to guarantee strictly work-related, grounded technical "
    "consultation across `LangGraph`, `n8n`, `Make.com`, `GoHighLevel`, `RAG`, `FastAPI`, `Spring Boot`, and `PostgreSQL`. "
    "Share your email anytime to schedule a discovery call!"
)

EXAMPLE_PROMPTS: list[str] = [
    "How can you automate our agency's lead qualification and CRM workflows using n8n and GoHighLevel?",
    "Tell me about your production RAG and LangGraph agent architecture and the 75% support response time reduction.",
    "What is your enterprise backend experience with Java/Spring Boot, Python/FastAPI, and fintech payment systems?",
    "I'd like to hire Ajhar for a custom AI automation project. My name is Alex Rivera and my email is alex@riveraventures.com.",
]

CUSTOM_CSS: str = """
.gradio-container {
    max-width: 1024px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
.message {
    line-height: 1.6 !important;
    font-size: 0.98rem !important;
}
footer {
    opacity: 0.75;
}
"""
