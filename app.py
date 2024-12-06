import gradio as gr
from models.translator import translate_text 

# Giao diện Gradio
def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("<h1>Automatic Translation System</h1>")
        gr.Markdown("### Provide a URL and select the target language for translation.")
        
        with gr.Row(equal_height=True):
            url_input = gr.Textbox(
                label="URL", 
                placeholder="Enter a URL",
                lines=1
            )
            language_input = gr.Radio(
                choices=["English", "Spanish", "Vietnamese", "German", "French"], 
                label="Select Target Language", 
                value="English"
            )
            submit_button = gr.Button("Translate")

        with gr.Row():
            original_text_output = gr.Textbox(
                label="Original Text", 
                placeholder="The original text will appear here.",
                interactive=False,
            )

            result_output = gr.Textbox(
                label="Translated Text",
                placeholder="The translated text will appear here.",
                interactive=False
            )
            
        submit_button.click(
            translate_text,
            inputs=[url_input, language_input],
            outputs=[original_text_output, result_output]
        )
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
