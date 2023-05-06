import gradio as gr
from searchgpt.pirate_search import respond


with gr.Blocks(title="SearchGPT") as demo:
    chatbot = gr.Chatbot(label="SearchGPT").style(height=400)
    query = gr.Textbox(show_label=False)
    query.submit(respond, [query, chatbot], [query, chatbot])

demo.launch(share=False)
