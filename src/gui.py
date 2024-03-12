import random
import gradio as gr
from rag_fusion import RagFusion

def random_response(message, history):
    return random.choice(["Yes", "No"])



gr.ChatInterface(random_response).launch()
