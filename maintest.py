import os
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# 測試 Gradio Group")
    with gr.Group():
        gr.Markdown("這是一個 Group 區塊")
        gr.Textbox(label="你的問題")
        gr.Button("送出")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Render 會給你一個 PORT 環境變數
    demo.launch(server_name="0.0.0.0", server_port=port)
