# app.py
"""
AI Mini-Detective — Single-file multi-agent CrewAI pipeline using Gemini (LangChain wrapper)
Fixed: build_agents_metadata now returns a list (not dict). Robust any_crew check.
"""
import os
import re
import json
import traceback
import streamlit as st
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List

# Prevent CrewAI import-time errors that expect OPENAI keys.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "DUMMY_PLACEHOLDER_OPENAI_KEY"

# Optional imports for Gemini / LangChain wrapper
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage
except Exception:
    ChatGoogleGenerativeAI = None
    HumanMessage = None

# Optional CrewAI imports (we create Agent metadata only if available)
try:
    from crewai import Agent
except Exception:
    Agent = None

load_dotenv()

# ------------------ CONFIG ------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5",
    "gemini-1.5-pro",
]

# Developer-provided header image path (will be transformed to hosted URL by platform)
HEADER_IMAGE_PATH = "/mnt/data/0136661b-fea6-4adf-befa-d57e380d7b04.png"

# Streamlit page
st.set_page_config(page_title="AI Mini-Detective — Multi-Agent (CrewAI + Gemini)", page_icon="🕵️", layout="centered")
st.markdown(
    """
    <style>
    .big-title {font-size:40px; font-weight:800; margin-bottom: 0.25rem;}
    .subtitle {font-size:15px; color: #6b7280; margin-top: 0.1rem; margin-bottom: 1.0rem;}
    .card {background:#ffffff;border-radius:10px;padding:18px;box-shadow: 0 6px 18px rgba(0,0,0,0.06);}
    </style>
    """,
    unsafe_allow_html=True,
)

if os.path.exists(HEADER_IMAGE_PATH):
    st.image(HEADER_IMAGE_PATH, use_column_width=True)

st.markdown('<div class="big-title">🕵️ AI Mini-Detective — Multi-Agent Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CrewAI-style agents, all powered by Gemini (no OpenAI)</div>', unsafe_allow_html=True)

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)
user_text = st.text_area("Describe your mystery here...", height=140, placeholder="e.g. I lost my USB drive after coming from class.")
use_crewai_checkbox = st.checkbox("Use full CrewAI multi-agent pipeline (Gemini-backed)", value=True)
run_button = st.button("Start Investigation 🔍")
st.markdown('</div>', unsafe_allow_html=True)

# session storage
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ------------------ Utilities & local fallback ------------------
def extract_clues_rule_based(text: str) -> Dict[str, Any]:
    t = text.strip()
    clues = {"key_objects": [], "locations": [], "people": [], "timeline": [], "other": []}
    # basic heuristics
    for match in re.finditer(r"\b(my|the|a)\s+([A-Za-z0-9\-\_ ]{2,30})(?:[.,;!?]|$)", t, re.I):
        candidate = match.group(2).strip()
        if len(candidate.split()) <= 4 and len(candidate) >= 2:
            clues["key_objects"].append(candidate)
    loc_keywords = ["cupboard", "class", "room", "library", "bus", "train", "campus", "home", "locker"]
    for k in loc_keywords:
        if re.search(rf"\b{k}\b", t, re.I):
            clues["locations"].append(k)
    if re.search(r"\bI\b|\bme\b|\bmyself\b", t):
        clues["people"].append("the speaker (I)")
    for w in ["teacher", "friend", "classmate", "security", "cleaner"]:
        if re.search(rf"\b{w}\b", t, re.I):
            clues["people"].append(w)
    if re.search(r"\bafter\b.*\b(class|lecture|coming)\b", t, re.I) or re.search(r"\bafter coming from\b", t, re.I):
        clues["timeline"].append("Lost after leaving class / after journey")
    if re.search(r"\blost\b|\bmissing\b", t, re.I):
        clues["other"].append("Described as 'lost' (likely misplacement)")
    for k in clues:
        clues[k] = list(dict.fromkeys(clues[k]))
    return clues

def local_analyze(text: str) -> Dict[str, Any]:
    clues = extract_clues_rule_based(text)
    h1 = {"title": "Accidental Drop During Transit", "reason": "Small items often fall during movement; unobserved drop while travelling.", "probability": 45}
    h2 = {"title": "Misplacement at Either End of the Journey", "reason": "Left in classroom or misplaced at destination and forgotten.", "probability": 55}
    recs = [
        "Search current location thoroughly (desks, bags, under furniture).",
        "Contact classroom staff / department lost & found and building services.",
        "Retrace the route taken after class, checking benches, stairwells, and transit areas.",
        "Post in local groups and check campus lost & found or security.",
        "If the USB contains sensitive data, consider changing passwords and enabling encryption/backups going forward.",
    ]
    return {"clues": clues, "hypotheses": [h1, h2], "conclusion": {"most_likely": h2["title"], "explanation": "Combined probability is slightly higher for misplacement at endpoints."}, "recommendations": recs}

# ------------------ Gemini (LangChain) + Adapter for CrewAI ------------------
class CrewaiGeminiAdapter:
    """
    Adapter exposing CrewAI-friendly methods, backed by LangChain ChatGoogleGenerativeAI (Gemini).
    We will call adapter.invoke(prompt_or_messages) to get a model response.
    """
    def __init__(self, gemini_llm):
        self._llm = gemini_llm

    def invoke(self, messages):
        if isinstance(messages, str):
            if HumanMessage is not None:
                return self._llm.invoke([HumanMessage(content=messages)])
            else:
                return self._llm.invoke([("human", messages)])
        return self._llm.invoke(messages)

    def generate(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return self._llm.generate(prompts)

    def stream(self, messages):
        return self._llm.stream(messages)

    async def ainvoke(self, messages):
        return await self._llm.ainvoke(messages)

    def __call__(self, prompt: str):
        return self.invoke(prompt)

    @staticmethod
    def extract_text(resp):
        if resp is None:
            return None
        if hasattr(resp, "content"):
            return resp.content
        if hasattr(resp, "generations"):
            gens = getattr(resp, "generations")
            try:
                cand = gens[0][0] if isinstance(gens[0], (list, tuple)) else gens[0]
                if hasattr(cand, "text"):
                    return cand.text
                if hasattr(cand, "content"):
                    return cand.content
            except Exception:
                pass
        try:
            return str(resp)
        except Exception:
            return None

def create_gemini_llm_try_models():
    if ChatGoogleGenerativeAI is None or not GOOGLE_API_KEY:
        return None, None
    attempts = []
    for model in GEMINI_MODEL_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(model=model, temperature=0.25, google_api_key=GOOGLE_API_KEY)
            return llm, model
        except Exception as e:
            attempts.append((model, str(e)))
    st.warning("Could not initialize Gemini with any candidate model. Falling back to local analyzer.")
    for m, e in attempts:
        st.text(f"{m}: {e[:300]}")
    return None, None

# ------------------ CrewAI multi-agent pipeline (orchestrated) ------------------
def build_agents_metadata() -> List[Dict[str, Any]]:
    """
    Return a list of agent metadata dicts; each dict contains:
      - id, role, goal, backstory, crew_agent (optional)
    This is intentionally a LIST to match how the orchestration iterates agents.
    """
    specs = [
        {
            "id": "evidence",
            "role": "Evidence Extractor",
            "goal": "Extract objective clues and facts from the case.",
            "backstory": "A forensic-minded investigator who prefers concise, factual bullet lists."
        },
        {
            "id": "scenario",
            "role": "Scenario Analyst",
            "goal": "Generate 2 plausible scenarios/hypotheses that explain the evidence.",
            "backstory": "A creative analyst converting facts into plausible alternative explanations."
        },
        {
            "id": "prob",
            "role": "Probability Assessor",
            "goal": "Assign probability scores and short rationales to each scenario.",
            "backstory": "A statistical pragmatist: honest, conservative, and explainable."
        },
        {
            "id": "conclude",
            "role": "Conclusion Agent",
            "goal": "Synthesize findings and provide clear next steps and a short conclusion.",
            "backstory": "A calm, practical senior detective who gives actionable guidance."
        }
    ]

    agents_list: List[Dict[str, Any]] = []
    for s in specs:
        crew_obj = None
        if Agent is not None:
            try:
                # we intentionally pass llm=None — calls go via our adapter
                crew_obj = Agent(role=s["role"], goal=s["goal"], backstory=s["backstory"], verbose=False, allow_delegation=False, llm=None)
            except Exception:
                crew_obj = None
        agents_list.append({
            "id": s["id"],
            "role": s["role"],
            "goal": s["goal"],
            "backstory": s["backstory"],
            "crew_agent": crew_obj
        })
    return agents_list

def run_multi_agent_pipeline_gemini(adapter: CrewaiGeminiAdapter, user_input: str) -> Dict[str, Any]:
    """
    Orchestrate the 4-agent pipeline by calling the adapter for each step.
    Returns a dict with structured outputs.
    """
    # 1) Evidence extraction
    evidence_prompt = (
        "You are an Evidence Extractor. Extract short, bullet-style clues from the case below. "
        "Return JSON with fields: key_objects (list), locations, people, timeline, other (list). "
        "Case:\n" + user_input
    )
    ev_resp = adapter.invoke(evidence_prompt)
    ev_text = adapter.extract_text(ev_resp) or ""
    clues = {}
    try:
        clues = json.loads(ev_text)
    except Exception:
        clues = extract_clues_rule_based(user_input)
        clues.setdefault("llm_note", ev_text[:1000])

    # 2) Scenario analyst: propose 2 scenarios
    scenario_prompt = (
        "You are a Scenario Analyst. Using the extracted clues below, propose 2 concise plausible scenarios/hypotheses. "
        "For each scenario, give: title, short_reason, suggested_evidence_to_check. Return JSON list of 2 objects.\n\n"
        f"Extracted clues: {clues}\n\n"
    )
    sc_resp = adapter.invoke(scenario_prompt)
    sc_text = adapter.extract_text(sc_resp) or ""
    scenarios = []
    try:
        scenarios = json.loads(sc_text)
    except Exception:
        scenarios = [
            {"title": "Accidental Drop During Transit", "short_reason": "Item likely fell from pocket/bag while moving.", "suggested_evidence_to_check": ["route benches", "stairs", "floor near doors"]},
            {"title": "Left Behind / Misplaced at endpoints", "short_reason": "Left in classroom or misplaced at destination and forgotten.", "suggested_evidence_to_check": ["class desks", "bag contents", "destination surfaces"]}
        ]

    # 3) Probability agent: assign probabilities and short rationale
    prob_prompt = (
        "You are a Probability Assessor. Given these scenarios and clues, assign integer probability percentages that sum to ~100 and give a one-line rationale for each. "
        "Return JSON list matching scenarios with fields: title, probability, rationale.\n\n"
        f"scenarios: {scenarios}\nclues: {clues}\n"
    )
    pr_resp = adapter.invoke(prob_prompt)
    pr_text = adapter.extract_text(pr_resp) or ""
    probabilities = []
    try:
        probabilities = json.loads(pr_text)
    except Exception:
        probabilities = [
            {"title": scenarios[0]["title"], "probability": 45, "rationale": "Transit drops are common."},
            {"title": scenarios[1]["title"], "probability": 55, "rationale": "Leaving items behind is slightly more likely overall."}
        ]

    # 4) Conclusion & Actions: finalize analysis and steps
    conclude_prompt = (
        "You are the Conclusion Agent. Synthesize the clues, scenarios, and probabilities into a clear, actionable report. "
        "Return a human-readable summary with sections: Key facts, Scenarios (with probabilities), Recommended next steps (bulleted), and Short conclusion.\n\n"
        f"Clues: {clues}\nScenarios: {scenarios}\nProbabilities: {probabilities}\n"
    )
    co_resp = adapter.invoke(conclude_prompt)
    co_text = adapter.extract_text(co_resp) or ""

    result = {
        "clues": clues,
        "scenarios": scenarios,
        "probabilities": probabilities,
        "conclusion_text": co_text
    }
    return result

# ------------------ Updated render_final_result (robust to strings) ------------------
def render_final_result(result: Any, model_name: Optional[str]):
    """
    Accepts either:
      - result: dict with keys 'clues','scenarios','probabilities','conclusion_text'
      - result: plain string (LLM free-form) — we'll show it as the conclusion/body
    """
    st.markdown("----")

    # If the LLM returned a plain string, render it as the main analysis text
    if isinstance(result, str):
        st.markdown("### 🕵️ Case Analysis (raw LLM output)")
        st.markdown(result)
        if model_name:
            st.markdown(f"\n*Analysis powered by Gemini model: `{model_name}` (raw output)*")
        else:
            st.markdown("\n*Analysis powered by local analyzer.*")
        return

    # If result is not a dict, try to stringify it safely
    if not isinstance(result, dict):
        st.markdown("### 🕵️ Case Analysis")
        try:
            st.markdown(str(result))
        except Exception:
            st.write(result)
        if model_name:
            st.markdown(f"\n*Analysis powered by Gemini model: `{model_name}`*")
        else:
            st.markdown("\n*Analysis powered by local analyzer.*")
        return

    # From here on 'result' is a dict as expected
    st.markdown("### 🔎 Extracted Clues")
    clues = result.get("clues", {}) or {}

    # If clues itself is a string (some LLMs return plain text), show that
    if isinstance(clues, str):
        st.write(clues)
    else:

        if clues.get("other"):
            st.markdown("**Other**")
            for o in clues.get("other", []):
                st.write(f"- {o}")

        if clues.get("llm_note"):
            st.markdown("**LLM Note**")
            st.write(clues.get("llm_note"))

    st.markdown("### 🧩 Scenarios")
    for s in result.get("scenarios", []) or []:
        if isinstance(s, dict):
            st.markdown(f"**{s.get('title') or 'Scenario'}**")
            if s.get("short_reason"):
                st.write(s.get("short_reason"))
            ev_checks = s.get("evidence_to_check") or s.get("suggested_evidence_to_check") or []
            if ev_checks:
                st.write("Check:", ", ".join(ev_checks))
        else:
            st.markdown(f"**{s}**")
        st.write("---")
    if not result.get("scenarios"):
        st.write("- (none generated)")

    st.markdown("### 📊 Probabilities")
    for p in result.get("probabilities", []) or []:
        if isinstance(p, dict):
            st.write(f"- **{p.get('title')}** — {p.get('probability')}% — {p.get('rationale')}")
        else:
            st.write(f"- {p}")
    if not result.get("probabilities"):
        st.write("- (none assigned)")

    st.markdown("### ✅ Conclusion & Recommended Next Steps")
    conclusion = result.get("conclusion_text") or result.get("conclusion") or ""
    if isinstance(conclusion, str) and conclusion.strip():
        st.markdown(conclusion)
    elif isinstance(conclusion, dict):
        for k, v in conclusion.items():
            st.markdown(f"**{k}**")
            st.write(v)
    else:
        st.write("- (no conclusion text)")

    if model_name:
        st.markdown(f"\n*Analysis powered by Gemini model: `{model_name}`*")
    else:
        st.markdown("\n*Analysis powered by local analyzer.*")

# ------------------ Robust main handler ------------------
if run_button:
    if not user_text or user_text.strip() == "":
        st.error("Please describe the mystery before you Investigate.")
    else:
        st.info("Running multi-agent investigation...")

        # Try to create Gemini LLM (LangChain) and adapter
        gemini_llm, model_name = create_gemini_llm_try_models()
        if gemini_llm is None or not use_crewai_checkbox:
            st.warning("Gemini/LangChain wrapper not available or CrewAI disabled — using local analyzer.")
            fallback = local_analyze(user_text)
            local_report = {
                "clues": fallback["clues"],
                "scenarios": fallback["hypotheses"],
                "probabilities": [
                    {"title": fallback["hypotheses"][0]["title"], "probability": fallback["hypotheses"][0]["probability"], "rationale": fallback["hypotheses"][0]["reason"]},
                    {"title": fallback["hypotheses"][1]["title"], "probability": fallback["hypotheses"][1]["probability"], "rationale": fallback["hypotheses"][1]["reason"]}
                ],
                "conclusion_text": f"Most likely: {fallback['conclusion']['most_likely']}\n\n{fallback['conclusion']['explanation']}"
            }
            render_final_result(local_report, model_name=None)
            st.session_state["last_result"] = ("local", local_report)
        else:
            adapter = CrewaiGeminiAdapter(gemini_llm)
            agents_meta = build_agents_metadata()  # now returns a list
            try:
                result = run_multi_agent_pipeline_gemini(adapter, user_text)
                render_final_result(result, model_name=model_name)
                st.session_state["last_result"] = ("gemini", result)

                # robust any_crew check — safe even if agents_meta is malformed
                any_crew = False
                try:
                    if isinstance(agents_meta, list):
                        for a in agents_meta:
                            if isinstance(a, dict) and a.get("crew_agent"):
                                any_crew = True
                                break
                    elif isinstance(agents_meta, dict):
                        # defensive: if a dict slipped through, check values
                        for v in agents_meta.values():
                            if hasattr(v, "role") or (isinstance(v, dict) and v.get("crew_agent")):
                                any_crew = True
                                break
                except Exception:
                    any_crew = False

                if any_crew:
                    st.caption("CrewAI Agent metadata created and bound (LLM calls routed via Gemini adapter).")
            except Exception:
                st.error("Multi-agent pipeline failed; showing fallback and traceback.")
                st.text(traceback.format_exc()[:2000])
                fallback = local_analyze(user_text)
                fallback_report = {
                    "clues": fallback["clues"],
                    "scenarios": fallback["hypotheses"],
                    "probabilities": [
                        {"title": fallback["hypotheses"][0]["title"], "probability": fallback["hypotheses"][0]["probability"], "rationale": fallback["hypotheses"][0]["reason"]},
                        {"title": fallback["hypotheses"][1]["title"], "probability": fallback["hypotheses"][1]["probability"], "rationale": fallback["hypotheses"][1]["reason"]}
                    ],
                    "conclusion_text": f"Most likely: {fallback['conclusion']['most_likely']}\n\n{fallback['conclusion']['explanation']}"
                }
                render_final_result(fallback_report, model_name=None)
                st.session_state["last_result"] = ("fallback", fallback_report)

# Show cached last result if present and no current run
elif st.session_state.get("last_result"):
    typ, payload = st.session_state["last_result"]
    st.markdown("### Last cached analysis")
    # payload may be dict or string
    render_final_result(payload, model_name="(cached gemini result)" if typ == "gemini" else None)
