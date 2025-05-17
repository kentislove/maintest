import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# 測試 Gradio Group")
    with gr.Group():
        gr.Markdown("這是一個 Group 區塊")
        gr.Textbox(label="你的問題")
        gr.Button("送出")

demo.launch()

