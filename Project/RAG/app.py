import gradio as gr
from rag_with_palm import RAGPaLMQuery

#palm
rag_palm_query_instance = RAGPaLMQuery()


chat_history = []

def chat_app(prompt):
    global chat_history
    

    chat_history.append({"role": "user", "message": prompt})


    conversation = "\n\n".join([f"{msg['role'].capitalize()}: {msg['message']}" for msg in chat_history])

    
    response = rag_palm_query_instance.query_response(conversation)

    
    chat_history.append({"role": "assistant", "message": response})

    return response


iface = gr.Interface(
    fn=chat_app,
    inputs=gr.Textbox(placeholder="Type your message here..."),
    outputs=gr.Textbox(),
    live=True,
    title="🤖 Chat with bot",
    css="""  # Add custom CSS for styling
        body {
            background-color: #f9f9f9;
            font-family: 'Arial', sans-serif;
            padding: 20px;
            display: flex;
            flex-direction: column;  # Change to column layout
            align-items: center;
        }
        .gr-form {
            max-width: 500px;
            margin-bottom: 20px;  # Change margin to bottom
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            width: 100%;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            max-width: 500px;
            width: 100%;
        }
        .user-message, .assistant-message {
            border-radius: 10px;
            margin-bottom: 10px;
            padding: 10px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #7cbcff;
            align-self: flex-end;
            color: #fff;
        }
        .assistant-message {
            background-color: #e6e6e6;
        }
        .gr-textbox {
            margin-top: 10px;  # Add margin to the textbox
        }
        .gr-live {
            margin-top: 10px;  # Add margin to the live button
        }
        .feedback-link {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;  # Add this line to remove underline
        }
    """
)


iface.launch()
