import gradio as gr

from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
import os


def loading_csv():
    return "Loading..."


def csv_changes(csv_doc, api_token):
    os.environ["OPENAI_API_KEY"] = api_token

    if api_token == "":
        return "API Key required!"

    global agent

    agent = create_csv_agent(
        OpenAI(temperature=0,
               ),
        csv_doc,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    return "Ready"


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response
    return history


def infer(question):
    result = agent.run(question)
    return result


css = """
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with csv</h1>
    <p style="text-align: center;">Upload a .csv from your computer, click the "Load csv to LangChain" button, <br />
    when everything is ready, you can start asking questions about the csv ;)</p>
</div>
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)

        with gr.Column():
            csv_doc = gr.File(label="Load a csv", file_types=[".csv"], type="filepath")
           
            api_token = gr.Textbox(
                label="Open AI API Token",
                placeholder="Enter your Open AI API Token",
            )
            with gr.Row():
                langchain_status = gr.Textbox(
                    label="Status", placeholder="", interactive=False
                )
                load_csv = gr.Button("Load csv to Langchain")

        chatbot = gr.Chatbot([], elem_id="chatbot")
        question = gr.Textbox(
            label="Question", placeholder="Type your question and hit Enter "
        )
        submit_btn = gr.Button("Send message")


    load_csv.click(
        csv_changes,
        inputs=[csv_doc, api_token],
        outputs=[langchain_status],
        queue=False,
    )
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )

demo.launch(debug=True)
