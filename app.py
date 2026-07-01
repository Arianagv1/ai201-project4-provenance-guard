#http://localhost:5000/dashboard
import os
import re
import json
import uuid
import sqlite3
import statistics

from flask import Flask, request, jsonify, render_template_string
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DB_PATH = "audit_log.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                content_id TEXT,
                creator_id TEXT,
                timestamp  TEXT,
                event TEXT DEFAULT 'classification',
                attribution TEXT,
                confidence REAL,
                llm_score REAL,
                stylo_score REAL,
                status TEXT,
                creator_reasoning TEXT
            )
        """)
        # Migrate databases created before the event / stylo_score /
        # creator_reasoning columns existed. ADD COLUMN with a constant default
        # backfills existing rows, so old classification rows get event=
        # 'classification' and remain discoverable by the appeal lookup.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)")}
        for col, decl in (
            ("event", "TEXT DEFAULT 'classification'"),
            ("stylo_score", "REAL"),
            ("creator_reasoning", "TEXT"),
        ):
            if col not in existing:
                conn.execute(f"ALTER TABLE audit_log ADD COLUMN {col} {decl}")

def log_event(entry):
    """Append one structured row to the audit log. Unspecified fields default to
    None so both classification and appeal events use the same writer."""
    row = {
        "content_id": None,
        "creator_id": None,
        "event": "classification",
        "attribution": None,
        "confidence": None,
        "llm_score": None,
        "stylo_score": None,
        "status": None,
        "creator_reasoning": None,
        **entry,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO audit_log "
            "(content_id, creator_id, timestamp, event, attribution, confidence, "
            " llm_score, stylo_score, status, creator_reasoning) VALUES "
            "(:content_id, :creator_id, :timestamp, :event, :attribution, "
            ":confidence, :llm_score, :stylo_score, :status, :creator_reasoning)",
            row,
        )

def get_classification(content_id):
    """Return the original classification row for a content_id, or None."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM audit_log WHERE content_id = ? AND event = 'classification' "
            "ORDER BY timestamp ASC LIMIT 1",
            (content_id,),
        ).fetchone()
    return dict(row) if row else None

def set_status(content_id, status):
    """Flip the content record's status (the classification row). Returns the
    number of rows updated."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "UPDATE audit_log SET status = ? WHERE content_id = ? AND event = 'classification'",
            (status, content_id),
        )
        return cur.rowcount

def read_log(limit=20):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]

def compute_analytics():
    """Aggregate the audit log into dashboard metrics. Computed from the same
    SQLite table both endpoints write to, so the numbers stay the single source
    of truth (planning.md). Returns a dict of three metrics:

      1. detection_pattern — count of each verdict (ai / human / uncertain) over
         classification events, plus the AI-vs-human ratio.
      2. appeal_rate — appeals filed per classification.
      3. mean_confidence_by_verdict — average confidence within each verdict, so
         a confident 'ai' call is distinguishable from a borderline one.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Metric 1 — detection pattern: verdict counts over classifications.
        verdict_rows = conn.execute(
            "SELECT attribution, COUNT(*) AS n FROM audit_log "
            "WHERE event = 'classification' GROUP BY attribution"
        ).fetchall()
        verdict_counts = {"ai": 0, "human": 0, "uncertain": 0}
        for row in verdict_rows:
            if row["attribution"] in verdict_counts:
                verdict_counts[row["attribution"]] = row["n"]

        total_classifications = sum(verdict_counts.values())
        n_appeals = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE event = 'appeal'"
        ).fetchone()[0]

        # Metric 3 — mean confidence per verdict.
        conf_rows = conn.execute(
            "SELECT attribution, AVG(confidence) AS avg_conf FROM audit_log "
            "WHERE event = 'classification' AND confidence IS NOT NULL "
            "GROUP BY attribution"
        ).fetchall()
        mean_conf = {row["attribution"]: row["avg_conf"] for row in conf_rows}

    # AI-vs-human ratio: AI verdicts per human verdict. None when there are no
    # human verdicts (division undefined) rather than a misleading 0.
    ai, human = verdict_counts["ai"], verdict_counts["human"]
    ai_vs_human_ratio = round(ai / human, 3) if human else None

    return {
        "total_classifications": total_classifications,
        "detection_pattern": {
            "counts": verdict_counts,
            "ai_share": round(ai / total_classifications, 3) if total_classifications else None,
            "human_share": round(human / total_classifications, 3) if total_classifications else None,
            "ai_vs_human_ratio": ai_vs_human_ratio,
        },
        "appeal_rate": {
            "appeals": n_appeals,
            "classifications": total_classifications,
            "rate": round(n_appeals / total_classifications, 3) if total_classifications else None,
        },
        "mean_confidence_by_verdict": {
            verdict: round(mean_conf.get(verdict), 3) if verdict in mean_conf else None
            for verdict in ("ai", "human", "uncertain")
        },
    }

# --- Signal 1: LLM-based classification (Groq) ---------------------------
# Asks the model to estimate P(AI) for a passage. Per planning.md, the LLM is
# the stronger holistic signal but is biased toward calling long/verbose text
# "AI"; the prompt explicitly tells it to ignore length and judge on style.

GROQ_MODEL = "llama-3.3-70b-versatile"

LLM_SYSTEM_PROMPT = (
    "You are a forensic text-attribution analyst. Given a passage, estimate "
    "the probability that it was generated by an AI language model rather than "
    "written by a human.\n\n"
    "Judge on stylistic and structural evidence: generic phrasing, unnaturally "
    "even rhythm, lack of an idiosyncratic voice, over-balanced structure, and "
    "hedging. Do NOT treat length, verbosity, vocabulary level, or topic as "
    "evidence on their own — short and long texts are equally likely to be "
    "human. Remember that genuine human writing often contains sarcasm, humor, "
    "irregular rhythm, and imperfection.\n\n"
    "Respond with ONLY a JSON object, no extra prose, in exactly this form:\n"
    '{"p_ai": <number between 0 and 1>, "reasoning": "<one short sentence>"}\n'
    "where p_ai is your probability the text is AI-generated "
    "(0 = certainly human, 1 = certainly AI)."
)

_groq_client = None

def get_groq_client():
    """Lazily create the Groq client so importing this module never requires
    a key (handy for testing the rest of the app offline)."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

def llm_signal(text):
    """Signal 1. Returns {"p_ai": float in [0,1], "reasoning": str}."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    data = json.loads(response.choices[0].message.content)
    p_ai = max(0.0, min(1.0, float(data["p_ai"])))  # clamp into [0, 1]
    return {"p_ai": p_ai, "reasoning": data.get("reasoning", "")}

def attribution_from_p_ai(p_ai):
    """Map P(AI) to a label using the asymmetric thresholds from planning.md."""
    if p_ai >= 0.75:
        return "ai"
    if p_ai <= 0.35:
        return "human"
    return "uncertain"


# --- Signal 2: Stylometric heuristics ------------------------------------
# Pure-Python signal (no external libs). Computes three metrics, turns each
# into an "AI-ness" sub-score in [0, 1], and takes a weighted combination.
# The phrase-marker metric is weighted highest because it is the only one that
# separates clearly-AI prose from formal-but-human prose (burstiness alone
# ranks uniform human writing as AI). Mapping constants are heuristic starting
# values, tuned with calibration data per planning.md.

# Words/phrases disproportionately favored by LLM-generated prose. Substring
# matching is intentional so inflections are caught ("navigate" -> "navigating").
AI_MARKERS = (
    "furthermore", "moreover", "additionally", "in addition", "consequently",
    "it is important to note", "it is worth noting", "in conclusion",
    "ultimately", "overall", "studies show", "research suggests",
    "there are several", "numerous", "various", "transformative", "paradigm",
    "leverage", "stakeholders", "essential", "crucial", "pivotal", "robust",
    "holistic", "underscore", "delve", "realm", "landscape", "navigate",
)

def _clamp(x):
    return max(0.0, min(1.0, x))

def _split_sentences(text):
    return [s for s in re.split(r"[.!?]+", text) if s.strip()]

def _words(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

def stylometric_signal(text):
    """Signal 2. Returns {"p_ai": float in [0,1], "metrics": {...}}."""
    words = _words(text)
    sentences = _split_sentences(text)
    n_words = len(words)

    # Guard: too little text for the statistics to mean anything. Returning a
    # neutral 0.5 keeps a short snippet from being falsely flagged (this is the
    # "small sample size" limitation called out in planning.md).
    if n_words < 10 or len(sentences) < 2:
        return {
            "p_ai": 0.5,
            "metrics": {"note": "insufficient sample (text too short)"},
        }

    # Metric 1 — Burstiness: coefficient of variation of sentence lengths.
    # Humans vary sentence length a lot (high CV); AI tends toward uniform,
    # medium-length sentences (low CV). LOW variance -> more AI.
    lengths = [len(_words(s)) for s in sentences]
    mean_len = statistics.mean(lengths)
    cv = (statistics.pstdev(lengths) / mean_len) if mean_len else 0.0
    ai_burstiness = _clamp(1.0 - cv / 0.6)

    # Metric 2 — AI-phrase markers: density of LLM-favored connectives/cliches.
    # This replaced type-token ratio, which did not discriminate (modern AI has
    # high lexical diversity, same as humans). MORE markers -> leans AI.
    text_lower = text.lower()
    marker_count = sum(text_lower.count(m) for m in AI_MARKERS)
    ai_markers = _clamp(marker_count / 3.0)

    # Metric 3 — Comma density: commas per sentence. AI favors balanced,
    # comma-rich constructions ("First, ... Second, ..."); terse human text
    # uses few. HIGH comma density -> leans AI.
    commas_per_sentence = text.count(",") / len(sentences)
    ai_commas = _clamp(commas_per_sentence / 2.0)

    p_ai = 0.6 * ai_markers + 0.25 * ai_burstiness + 0.15 * ai_commas
    return {
        "p_ai": p_ai,
        "metrics": {
            "sentence_length_cv": round(cv, 3),
            "ai_marker_count": marker_count,
            "commas_per_sentence": round(commas_per_sentence, 3),
            "ai_burstiness": round(ai_burstiness, 3),
            "ai_markers": round(ai_markers, 3),
            "ai_commas": round(ai_commas, 3),
        },
    }


# --- Confidence scoring: fuse both signals -------------------------------
# Matches planning.md: weighted average (LLM stronger), confidence as distance
# from the 0.5 boundary, and the asymmetric attribution thresholds.

LLM_WEIGHT = 0.6
STYLO_WEIGHT = 0.4

def combine_signals(llm_p, stylo_p):
    """Fuse the two P(AI) scores into a final decision.
    Returns {"p_ai", "confidence", "attribution"}."""
    final_p = LLM_WEIGHT * llm_p + STYLO_WEIGHT * stylo_p
    confidence = 2.0 * abs(final_p - 0.5)   # distance from boundary, in [0,1]
    return {
        "p_ai": final_p,
        "confidence": confidence,
        "attribution": attribution_from_p_ai(final_p),
    }


# --- Transparency label: plain-language text the reader sees -------------
# Labels vary on BOTH direction (ai / human / uncertain) AND confidence, so a
# borderline call ("likely") reads differently from a sure one ("confident").
# This diverges from planning.md, which keyed the label on direction alone; the
# fix keeps the wording honest (a P(AI)=0.75 call is not something to be
# "confident" about). See the README spec reflection.
CONFIDENT_TIER = 0.70  # confidence >= this reads "confident"; below it, "likely"

TRANSPARENCY_LABELS = {
    ("ai", "confident"): "We are confident this text was generated by AI.",
    ("ai", "likely"): "This text was likely generated by AI, but we are not certain.",
    ("human", "confident"): "We are confident this text was written by a human.",
    ("human", "likely"): "This text was likely written by a human, but we are not certain.",
    "uncertain": "We can't confidently tell whether this text is AI-generated or human-written.",
}

def transparency_label(attribution, confidence):
    if attribution == "uncertain":
        return TRANSPARENCY_LABELS["uncertain"]
    tier = "confident" if confidence >= CONFIDENT_TIER else "likely"
    return TRANSPARENCY_LABELS[(attribution, tier)]


app = Flask(__name__)
init_db()

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.route("/")
def home():
    return "Provenance Guard is running."

@app.route("/log", methods=["GET"])
def view_log():
    return jsonify({"entries": read_log()})

@app.route("/analytics", methods=["GET"])
def analytics():
    """Dashboard metrics as JSON — auditable/testable, same read-only pattern as
    /log. The /dashboard view renders these same numbers."""
    return jsonify(compute_analytics())

# Rendered browser view over the exact numbers /analytics returns. Kept as an
# inline template so the dashboard is self-contained (no templates/ dir needed).
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Provenance Guard — Analytics</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; color: #1a1a1a; }
    h1 { margin-bottom: 0.25rem; }
    .sub { color: #666; margin-top: 0; }
    .cards { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
    .card { flex: 1 1 220px; border: 1px solid #e2e2e2; border-radius: 10px; padding: 1rem 1.25rem; }
    .card h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; color: #666; margin: 0 0 0.5rem; }
    .big { font-size: 2rem; font-weight: 700; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
    td, th { text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid #eee; font-size: 0.95rem; }
    .muted { color: #999; }
    .ai { color: #b3261e; } .human { color: #1b7a3d; } .uncertain { color: #8a6d00; }
  </style>
</head>
<body>
  <h1>Provenance Guard — Analytics</h1>
  <p class="sub">{{ a.total_classifications }} classification{{ '' if a.total_classifications == 1 else 's' }} logged</p>

  <div class="cards">
    <div class="card">
      <h2>Detection pattern</h2>
      <div class="big">
        {{ '%.2f'|format(a.detection_pattern.ai_vs_human_ratio) if a.detection_pattern.ai_vs_human_ratio is not none else '—' }}
      </div>
      <div class="muted">AI verdicts per human verdict</div>
    </div>
    <div class="card">
      <h2>Appeal rate</h2>
      <div class="big">
        {{ '%.0f%%'|format(a.appeal_rate.rate * 100) if a.appeal_rate.rate is not none else '—' }}
      </div>
      <div class="muted">{{ a.appeal_rate.appeals }} appeal{{ '' if a.appeal_rate.appeals == 1 else 's' }} / {{ a.appeal_rate.classifications }} classifications</div>
    </div>
  </div>

  <h2>Verdict breakdown</h2>
  <table>
    <tr><th>Verdict</th><th>Count</th><th>Share</th><th>Mean confidence</th></tr>
    {% for v in ['ai', 'human', 'uncertain'] %}
    <tr>
      <td class="{{ v }}">{{ v }}</td>
      <td>{{ a.detection_pattern.counts[v] }}</td>
      <td>
        {% if v == 'ai' %}{{ '%.0f%%'|format(a.detection_pattern.ai_share * 100) if a.detection_pattern.ai_share is not none else '—' }}
        {% elif v == 'human' %}{{ '%.0f%%'|format(a.detection_pattern.human_share * 100) if a.detection_pattern.human_share is not none else '—' }}
        {% else %}<span class="muted">—</span>{% endif %}
      </td>
      <td>
        {% if a.mean_confidence_by_verdict[v] is not none %}{{ '%.3f'|format(a.mean_confidence_by_verdict[v]) }}
        {% else %}<span class="muted">n/a</span>{% endif %}
      </td>
    </tr>
    {% endfor %}
  </table>

  <p class="sub" style="margin-top:1.5rem;">Raw numbers at <a href="/analytics">/analytics</a> · audit log at <a href="/log">/log</a></p>
</body>
</html>
"""

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Human-facing view of the three metrics, rendered from compute_analytics()."""
    return render_template_string(DASHBOARD_TEMPLATE, a=compute_analytics())

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    creator_id = data.get("creator_id")

    # Minimum required fields.
    if not text or not creator_id:
        return jsonify({"error": "Both 'text' and 'creator_id' are required."}), 400

    # Signal 1: LLM-based classification (Groq).
    try:
        signal_1 = llm_signal(text)
    except Exception as exc:  # noqa: BLE001 - surface upstream failures cleanly
        return jsonify({"error": f"Detection signal failed: {exc}"}), 502

    # Signal 2: Stylometric heuristics (pure Python, no upstream call).
    signal_2 = stylometric_signal(text)

    # Fuse the two P(AI) scores into a final P(AI), confidence, and direction,
    # then map the direction to the reader-facing transparency label.
    decision = combine_signals(signal_1["p_ai"], signal_2["p_ai"])
    attribution = decision["attribution"]
    confidence = decision["confidence"]
    label = transparency_label(attribution, confidence)

    content_id = str(uuid.uuid4())

    log_event({
        "content_id": content_id,
        "creator_id": creator_id,
        "event": "classification",
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": signal_1["p_ai"],
        "stylo_score": signal_2["p_ai"],
        "status": "classified",
    })

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": round(confidence, 3),
        "label": label,
        # Pipeline detail, exposed so the fused decision is auditable:
        "p_ai": round(decision["p_ai"], 3),
        "llm_score": round(signal_1["p_ai"], 3),
        "llm_reasoning": signal_1["reasoning"],
        "stylo_score": round(signal_2["p_ai"], 3),
        "stylo_metrics": signal_2["metrics"],
    })

@app.route("/appeal", methods=["POST"])
@limiter.limit("10 per minute")
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    reasoning = data.get("creator_reasoning")

    if not content_id or not reasoning:
        return jsonify(
            {"error": "Both 'content_id' and 'creator_reasoning' are required."}
        ), 400

    # The appeal must reference a real prior decision so we can log it "side by
    # side" with the original (planning.md, Appeals Workflow).
    original = get_classification(content_id)
    if original is None:
        return jsonify(
            {"error": f"No classification found for content_id '{content_id}'."}
        ), 404

    # 1. Flip the content record's status to "under review".
    set_status(content_id, "under review")

    # 2. Append the appeal to the audit log, carrying the creator's reasoning and
    #    a copy of the original decision's scores so the two sit side by side.
    #    No automated re-classification — a human reviews from here.
    log_event({
        "content_id": content_id,
        "creator_id": original["creator_id"],
        "event": "appeal",
        "attribution": original["attribution"],
        "confidence": original["confidence"],
        "llm_score": original["llm_score"],
        "stylo_score": original["stylo_score"],
        "status": "under review",
        "creator_reasoning": reasoning,
    })

    return jsonify({
        "content_id": content_id,
        "status": "under review",
        "message": "Your appeal was received and is under review.",
        "original_decision": {
            "attribution": original["attribution"],
            "confidence": original["confidence"],
            "llm_score": original["llm_score"],
            "stylo_score": original["stylo_score"],
        },
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
