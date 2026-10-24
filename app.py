"""Browser demo: speak into the mic (or type) and hear ZentroMart's reply.  python app.py"""
import gradio as gr

from src import config
from src.agent import SupportAgent
from src.voice import RECOVERY_REPLY, Voice, detect_lang

agent = SupportAgent()
voice = Voice()


def turn(audio, text, history):
    history = history or []
    lang = config.DEFAULT_LANG
    heard = (text or "").strip()
    if not heard and audio:
        heard = voice.transcribe(audio, lang)
    res = agent.respond_full(heard) if heard else None
    reply = res.reply if res else RECOVERY_REPLY
    wav = voice.speak(reply, config.ROOT / "reply.wav", detect_lang(reply, lang))
    history += [{"role": "user", "content": heard or "(couldn't hear anything)"},
                {"role": "assistant", "content": reply}]
    info = ""
    if res:
        info = (f"decision: {res.decision} | tools: {[c['action'] for c in res.tool_calls]} | "
                f"sources: {[c['id'] for c in res.chunks[:4]]}")
    return history, wav, info, None, ""


def reset():
    agent.reset()
    return [], None, "", None, ""


with gr.Blocks(title="ZentroMart Voice Support") as demo:
    gr.Markdown("## ZentroMart voice support agent\nRecord your question, or type it. Hear the reply.")
    chat = gr.Chatbot(type="messages", height=360)
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
    box = gr.Textbox(label="...or type", placeholder="My order hasn't arrived. Can I cancel it?")
    out = gr.Audio(label="Agent voice", autoplay=True)
    info = gr.Markdown()
    with gr.Row():
        go = gr.Button("Send", variant="primary")
        new = gr.Button("New customer")
    go.click(turn, [mic, box, chat], [chat, out, info, mic, box])
    new.click(reset, None, [chat, out, info, mic, box])

if __name__ == "__main__":
    demo.launch()
