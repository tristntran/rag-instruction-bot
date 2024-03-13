import random
import gradio as gr
from rag_fusion import RagFusion

def random_response(message, history):
    return random.choice(["Yes", "No"])


if __name__ == "__main__":
    gr.ChatInterface(random_response).launch()
