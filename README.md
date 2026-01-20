# Agentic A/B Experimentation Assistant

This project implements a multi-agent A/B experimentation assistant that mirrors
how product data scientists work at large tech companies. It translates natural
language experiment descriptions into formal hypotheses, analyzes experimental
results using statistical testing, and produces ship/no-ship recommendations with
a generated experiment report.

## Architecture

- Experiment Designer Agent (LLM): converts product ideas into experiment plans
- Data Analyst Agent (Python): computes uplift, significance, and plots
- Decision Agent (LLM): weighs statistical and practical significance
- Report Writer Agent (LLM): generates a concise markdown experiment report

## How to Run

1. Create a virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Create a .env file:
   OPENAI_API_KEY=your_key_here
4. Run:
   python -m src.run_demo

## Limitations and Future Work
- Add guardrail metrics (latency, errors)
- Segment analysis by device and geography
- Power and sample size estimation
- Streamlit UI for PMs
