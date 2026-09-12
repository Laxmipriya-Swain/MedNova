from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv
from markupsafe import Markup
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

# Lazy-load QA chain on first request to prevent OOM on Render free tier
_qa_chain = None
_qa_chain_loaded = False

def get_qa_chain():
    global _qa_chain, _qa_chain_loaded
    if not _qa_chain_loaded:
        from app.components.retriever import create_qa_chain
        _qa_chain = create_qa_chain()
        _qa_chain_loaded = True
    return _qa_chain

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
                qa_chain = get_qa_chain()  # ✅ Lazy-load on first request
                if qa_chain is None:
                    raise Exception("QA CHAIN could not be created (llm or vectorstore issue)")

                    result = qa_chain.invoke(user_input)

            messages.append({"role": "assistant", "content": result})
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