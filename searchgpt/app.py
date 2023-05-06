import gradio as gr


def respond(message, history):
    history.append((message, str.upper(message)))
    return "", history


with gr.Blocks(title="SearchGPT") as demo:
    chatbot = gr.Chatbot(label="SearchGPT").style(height=400)
    query = gr.Textbox(show_label=False)
    query.submit(respond, [query, chatbot], [query, chatbot])

demo.launch(share=False)
