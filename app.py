import pinecone
import promptlayer
import os
import gradio as gr
import re

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


# LOCAL STUFF
# Set Key Variables
from dotenv import load_dotenv
load_dotenv()
promptlayer.api_key = os.getenv("PROMPTLAYER_API_KEY")
pinecone.api_key = os.getenv("PINECONE_API_KEY")
# END LOCAL STUFF

# Prompt Template & Messages
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Based on the context provided, provide an answer to the best of your knowledge.
Use your skills to determine what kind of context is provided and tailor your response accordingly. 
When providing an answer, choose the tone of voice and humor of Zapp Brannigan from Futurama. Also, use html bullet list format when needed.
Question: {question}
=========
{context}
=========
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

# Initialize Pinecone
pinecone.init(
    environment="us-central1-gcp"
)
index_name = "support-kb"

# Replace kb_db_store initialization with Pinecone.from_existing_index method
embeddings = OpenAIEmbeddings()
kb_db = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="lsvt")


# Third, we need to create the prompt
# Initialize the ChatVectorDBChain
kb_chat = ConversationalRetrievalChain.from_llm(
    PromptLayerChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", pl_tags=["local", "customTemp7"]),
    retriever=kb_db.as_retriever(search_kwargs={"k": 3}),    
    verbose=True, 
    return_source_documents=True,
    qa_prompt=QA_PROMPT,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
)

# Format the text in the return
def format_terms(text):
    formatted_text = re.sub(r'[“"”]([^”“]+)[“"”]', r'<b>\1</b>', text)
    formatted_text = formatted_text.replace('\n', '<br>')
    return formatted_text


chat_history = []
def get_answer(query):
    
    result = kb_chat({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"])) 

     # Use a set to remove duplicates
    source_urls = list(set(doc.metadata["source"] for doc in result["source_documents"]))  
    
    formatted_answer = format_terms(result['answer'])    
    
    answer = f"<strong>Answer:</strong><br><br> {formatted_answer}<br><br><strong>Source URLs:</strong><br>"
    answer += "<br>".join(f'<a href="{url}" target="_blank">{url}</a>' for url in source_urls)
    
    return answer

# Gradio Interface
def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    query = history[-1][0]
    result = get_answer(query)
    history[-1][1] = result
    return history

# Launch the Gradio interface with the chatbot components and functions
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()

    with gr.Row():
        msg = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=2,
        )
        submit = gr.Button(value="Submit", variant="primary").style(full_width=False)
        clear = gr.Button("Clear", style="secondary").style(full_width=False)

    gr.Examples(
        examples = [
            ["How do I create a new user?"],
            ["what's new in the past month?"],
            ["what was released in October?"],
            ["how do I add files to the media library?"],
            ["what is a postable?"],
            ["what formats and resolutions do you support in the media library?"],
            ["how do I configure a certification?"]
        ],
        inputs=msg,
    )

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Row():
        msg
        submit
        clear

demo.launch()