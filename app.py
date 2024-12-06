import gradio as gr
from model.translator import translate_text

with gr.Blocks() as demo:
    gr.Markdown("<h1>Please select a language and provide a URL.</h1>")
    with gr.Row(equal_height=True):
        url = gr.Textbox(label="URL")
        language = gr.Radio(
            choices=["English", "Spanish", "Vietnamese", "German", "French"], 
            label="Language", 
            value="English",
            scale=0.5
        )
        button = gr.Button("Submit",scale=0.5)
    textbox = gr.Textbox(label="Result")

    button.click(
        translate_text,
        inputs=[url, language],
        outputs=textbox
    )

demo.launch()
