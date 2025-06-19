import gradio as gr
import streamlit as st
from model_loader import load_model
from rag import load_vectorstore

tokenizer, model = load_model()
vectorstore = load_vectorstore()

def chatbot_response(user_input):
    docs = vectorstore.similarity_search_with_score(user_input, k=5)  # Get scores
    sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)  # Sort by relevance score
    best_matches = [doc[0].page_content for doc in sorted_docs[:3]]  # Get top 3
    response = "\n".join(best_matches) if best_matches else "No relevant results found."
    return response

# 🟢 Gradio UI
def gradio_ui():
    demo = gr.Interface(fn=chatbot_response, inputs="text", outputs="text")
    demo.launch()

# 🔵 Streamlit UI
def streamlit_ui():
    st.title("Gen AI Studio Lite")
    user_input = st.text_input("Enter your query:")
    if user_input:
        response = chatbot_response(user_input)
        st.write(response)

if __name__ == "__main__":
    gradio_ui()  # Switch to streamlit_ui() if needed