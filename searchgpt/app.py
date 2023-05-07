import gradio as gr
from searchgpt.llm import respond


with gr.Blocks(title="SearchGPT") as demo:
    chatbot = gr.Chatbot(label="SearchGPT").style(height=400)
    query = gr.Textbox(label="Your question")
    query.submit(respond, [query, chatbot], [query, chatbot])

demo.launch(share=False)
