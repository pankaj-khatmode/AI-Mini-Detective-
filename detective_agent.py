# app.py
import os
import re
import traceback
import streamlit as st
from dotenv import load_dotenv
from typing import Any, Dict

# Optional imports for Gemini / LangChain wrapper
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage
except Exception:
    ChatGoogleGenerativeAI = None
    HumanMessage = None

load_dotenv()

# ------------------ CONFIG ------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # set in .env if you want Gemini
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5",
    "gemini-1.5-pro",  # legacy fallback if your account still supports it
]

# Use uploaded header image path provided earlier
HEADER_IMAGE_PATH = "/mnt/data/0136661b-fea6-4adf-befa-d57e380d7b04.png"

# Page setup
st.set_page_config(page_title="AI Mini-Detective — Micro Investigation Game", page_icon="🕵️", layout="centered")
st.markdown(
    """
    <style>
    .big-title {font-size:44px; font-weight:800; margin-bottom: 0.25rem; color: var(--text-color);}
    .subtitle {font-size:18px; color: #8b98a5; margin-top: 0.1rem; margin-bottom: 1.2rem;}
    .card {background:#0f1724;border-radius:10px;padding:18px;box-shadow: 0 6px 30px rgba(0,0,0,0.45); color:var(--text-color);}
    .investigate-btn {background: #0ea5a4; color:white; padding:10px 18px; border-radius:8px; border:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header image
if os.path.exists(HEADER_IMAGE_PATH):
    st.image(HEADER_IMAGE_PATH, use_column_width=True)

st.markdown('<div class="big-title">🕵️ AI Mini-Detective — Micro Investigation Game</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Crack the case with AI-powered investigation!</div>', unsafe_allow_html=True)

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)
user_text = st.text_area("Describe your mystery here...", height=160, placeholder="e.g. my phone / keys / pet is missing...")
investigate = st.button("Investigate 🔍")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session_state container for last result
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ------------------ Rule-based clue extraction (fallback) ------------------
def extract_clues_rule_based(text: str) -> Dict[str, Any]:
    t = text.strip()
    clues = {"key_objects": [], "locations": [], "people": [], "timeline": [], "other": []}

    # key objects heuristic (my X / the X)
    for match in re.finditer(r"\b(my|the|a)\s+([A-Za-z0-9\-\_ ]{2,30})(?:[.,;!?]|$)", t, re.I):
        candidate = match.group(2).strip()
        # filter out short/irrelevant words
        if len(candidate.split()) <= 4 and len(candidate) >= 2:
            clues["key_objects"].append(candidate)

    # locations heuristic
    loc_keywords = ["cupboard", "class", "room", "library", "bus", "train", "campus", "home", "locker"]
    for k in loc_keywords:
        if re.search(rf"\b{k}\b", t, re.I):
            clues["locations"].append(k)

    # people heuristic
    if re.search(r"\bI\b|\bme\b|\bmyself\b", t):
        clues["people"].append("the speaker (I)")
    for w in ["teacher", "friend", "classmate", "security", "cleaner"]:
        if re.search(rf"\b{w}\b", t, re.I):
            clues["people"].append(w)

    # timeline
    if re.search(r"\bafter\b.*\b(class|lecture|coming)\b", t, re.I) or re.search(r"\bafter coming from\b", t, re.I):
        clues["timeline"].append("Lost after leaving class / after journey")

    if re.search(r"\blost\b|\bmissing\b", t, re.I):
        clues["other"].append("Described as 'lost' (likely misplacement)")

    # dedupe lists
    for k in clues:
        clues[k] = list(dict.fromkeys(clues[k]))
    return clues

# ------------------ Local analysis (fallback) ------------------
def local_analyze(text: str) -> Dict[str, Any]:
    clues = extract_clues_rule_based(text)
    h1 = {
        "title": "Accidental Drop During Transit",
        "reason": (
            "Small items frequently fall during movement; the timeline states the loss happened after leaving class, "
            "which is consistent with an unnoticed drop while travelling."
        ),
        "probability": 45,
    }
    h2 = {
        "title": "Misplacement at Either End of the Journey",
        "reason": (
            "Items are commonly left on desks or placed down at the destination and forgotten. "
            "Given the loss was realized after arrival, this is slightly more likely overall."
        ),
        "probability": 55,
    }
    recs = [
        "Thoroughly search the current location (desks, bags, under furniture).",
        "Contact classroom staff / department lost & found and building services.",
        "Retrace the route taken after class, checking benches, stairwells, and transit areas.",
        "Post in local groups and check campus lost & found or security.",
        "If the USB contains sensitive data, consider changing passwords and enabling encryption/backups going forward.",
    ]
    return {"clues": clues, "hypotheses": [h1, h2], "conclusion": {"most_likely": h2["title"], "explanation": "Combined probability is slightly higher for misplacement at endpoints."}, "recommendations": recs}

# ------------------ Gemini helpers (best-effort) ------------------
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
    # no model worked
    st.warning("Could not initialize Gemini with any candidate model. Falling back to local analyzer.")
    for m, e in attempts:
        st.text(f"{m}: {e[:300]}")
    return None, None

def call_gemini_and_get_text(llm, prompt: str) -> str:
    try:
        if HumanMessage is not None:
            resp = llm.invoke([HumanMessage(content=prompt)])
        else:
            # fallback message tuple
            resp = llm.invoke([("system", "You are a detective."), ("human", prompt)])
        # try common shapes
        if hasattr(resp, "content"):
            return resp.content
        if hasattr(resp, "generations"):
            gens = getattr(resp, "generations")
            try:
                # try nested fields
                return str(gens[0][0].text) if (isinstance(gens, (list, tuple)) and gens and isinstance(gens[0], (list, tuple)) and hasattr(gens[0][0], "text")) else str(resp)
            except Exception:
                pass
        return str(resp)
    except Exception as e:
        st.text("Gemini invocation failed: " + str(e)[:300])
        st.text(traceback.format_exc()[:2000])
        return None

# ------------------ UI result rendering ------------------
def display_structured_result(result: Dict[str, Any], model_name: str = None):
    st.markdown("----")
    st.markdown("### 🔎 Extracted Clues")
    st.markdown("Here are the relevant clues extracted from the mystery:")
    clues = result["clues"]
    st.markdown("**Key objects mentioned**")
    if clues["key_objects"]:
        for obj in clues["key_objects"]:
            st.write(f"- {obj}")
    else:
        st.write("- (none identified)")

    st.markdown("**Locations**")
    if clues["locations"]:
        for loc in clues["locations"]:
            st.write(f"- {loc}")
    else:
        st.write("- (none identified)")

    st.markdown("**People/suspects**")
    if clues["people"]:
        for p in clues["people"]:
            st.write(f"- {p}")
    else:
        st.write("- (none identified)")

    st.markdown("**Timeline of events**")
    if clues["timeline"]:
        for t in clues["timeline"]:
            st.write(f"- {t}")
    else:
        st.write("- (timeline not explicit)")

    if clues["other"]:
        st.markdown("**Any other relevant information**")
        for o in clues["other"]:
            st.write(f"- {o}")

    st.markdown("### 🧩 Possible Scenarios")
    for h in result["hypotheses"]:
        st.markdown(f"**{h['title']}**")
        st.write(h["reason"])
        st.write(f"**Probability Rating:** {h['probability']}%")
        st.write("---")

    st.markdown("### 🎯 Analysis & Conclusion")
    st.markdown(f"**Most Likely Scenario:** {result['conclusion']['most_likely']}")
    st.write(result["conclusion"]["explanation"])

    st.markdown("### Recommended Next Steps")
    for r in result["recommendations"]:
        st.write(f"- {r}")

    if model_name:
        st.markdown(f"\n*Analysis powered by Gemini model: `{model_name}`*")
    else:
        st.markdown("\n*Analysis powered by local rule-based analyzer.*")

# ------------------ Robust main handler ------------------
if investigate:
    if not user_text or user_text.strip() == "":
        st.error("Please describe the mystery before you Investigate.")
    else:
        try:
            st.info("🔎 Investigating — please wait...")

            # Try Gemini (best-effort)
            gemini_llm, model_used = create_gemini_llm_try_models()
            gemini_response = None

            if gemini_llm:
                prompt = (
                    "You are a seasoned detective. Analyze the following short case and return a structured analysis.\n\n"
                    "Case description:\n" + user_text +
                    "\n\nReturn sections: Extracted Clues (Key objects, Locations, People, Timeline, Other), "
                    "2 Hypotheses with short reason and probability percentages, Conclusion (most likely scenario + explanation), "
                    "Recommended Next Steps (bulleted). Keep output clear and human-readable."
                )
                try:
                    gemini_response = call_gemini_and_get_text(gemini_llm, prompt)
                except Exception:
                    st.warning("Gemini call raised an exception; see traceback below.")
                    st.text(traceback.format_exc()[:2000])
                    gemini_response = None

            # Always compute local result so UI shows structured output
            local_result = local_analyze(user_text)

            # Present results: prefer Gemini text if available, else local
            if gemini_response:
                st.markdown("----")
                st.markdown("### 🕵️ Case Analysis (Gemini)")
                st.markdown(gemini_response)
                st.markdown("----")
                display_structured_result(local_result, model_name=model_used)
                st.session_state["last_result"] = ("gemini", gemini_response)
            else:
                display_structured_result(local_result, model_name=None)
                st.session_state["last_result"] = ("local", local_result)

            st.markdown("----")
            st.success("🕵️‍♂️ Case Status: Investigation Complete")

        except Exception:
            st.error("An unexpected error occurred while processing the request; traceback below:")
            st.text(traceback.format_exc()[:2000])
            try:
                fallback = local_analyze(user_text)
                display_structured_result(fallback)
                st.session_state["last_result"] = ("fallback", fallback)
            except Exception:
                st.text("Fallback analysis also failed; check the full traceback above.")

# If user hasn't pressed Investigate but we have a last_result, show it
elif st.session_state.get("last_result"):
    # helpful to re-render the last result on rerun
    typ, payload = st.session_state["last_result"]
    st.markdown("### Last analysis (cached)")
    if typ == "gemini":
        st.markdown("#### Gemini output (cached)")
        st.markdown(payload)
    else:
        st.markdown("#### Structured analysis (cached)")
        display_structured_result(payload)
