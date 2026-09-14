from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from dotenv import load_dotenv
from markupsafe import Markup
import os
import threading

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

# ── QA Chain (thread-safe eager warm-up) ────────────────────────────────────
_qa_chain = None
_qa_chain_lock = threading.Lock()
_qa_chain_ready = threading.Event()   # signals that warm-up finished

def _warmup_qa_chain():
    """Run in a background thread at startup so the first request is instant."""
    global _qa_chain
    try:
        from app.components.retriever import create_qa_chain
        chain = create_qa_chain()
        with _qa_chain_lock:
            _qa_chain = chain
    except Exception as e:
        # Log but don't crash the server — get_qa_chain() will surface the error
        print(f"[WARN] QA chain warm-up failed: {e}")
    finally:
        _qa_chain_ready.set()   # unblock any waiting requests

# Kick off warm-up immediately (non-blocking)
threading.Thread(target=_warmup_qa_chain, daemon=True, name="qa-warmup").start()

def get_qa_chain(timeout: int = 60):
    """Return the QA chain, waiting up to *timeout* seconds for warm-up."""
    _qa_chain_ready.wait(timeout=timeout)
    with _qa_chain_lock:
        return _qa_chain

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """
    Health-check endpoint.
    Use an uptime monitor (e.g. UptimeRobot, cron-job.org) to ping this URL
    every 10–14 minutes to prevent Render free-tier from spinning down.
    """
    ready = _qa_chain_ready.is_set()
    with _qa_chain_lock:
        chain_ok = _qa_chain is not None
    status = "ready" if (ready and chain_ok) else "warming_up"
    return jsonify({"status": status}), 200 if chain_ok else 503


@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session['messages']
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                qa_chain = get_qa_chain()

                if qa_chain is None:
                    raise Exception(
                        "QA CHAIN could not be created "
                        "(llm or vectorstore issue)"
                    )

                result = qa_chain.invoke(user_input)
                messages.append({
                    "role": "assistant",
                    "content": result
                })
                session["messages"] = messages

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return render_template(
                    "index.html",
                    messages=session["messages"],
                    error=error_msg
                )
        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))


@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
