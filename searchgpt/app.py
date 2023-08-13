import gradio as gr
from searchgpt.llm import respond


with gr.Blocks(title="SearchGPT") as demo:
    chatbot = gr.Chatbot(label="SearchGPT", height=400)
    query = gr.Textbox(label="Your question", autofocus=True)
    query.submit(respond, [query, chatbot], [query, chatbot])

demo.launch(share=False)
