# Talk to a document 📃🦜

This repository is for an app that lets you (a) summarize a document and (b) ask questions about the document. You can point to the URL that contains some text (e.g., an article, a blog post, or an essay) and the app will summarize its content. You can then chat with an agent to ask questions about the document.

## 🚀 Instructions to Launch the App

<details><summary><b>Show instructions</b></summary>

Once you make a copy of this codebase on your computer, activate a Python virtual environment using the following command:

`python -m venv .venv --prompt doc-qa`

Once the Python virtual environment is created, activate it and install all dependencies from `requirements.txt`.

`source .venv/bin/activate`

`pip install -r requirements.txt`

Once all dependencies are installed, you can launch the app using the following command:

`streamlit run src/app.py`

In a few seconds the app will be lanuched in your browser. If that doesn't happen automatically, you can copy the URL that's printed in the output.

</details>

## 🔑Secrets

<details><summary><b>Show config settings</b></summary>

This app makes a call to the OpenAI API. You will need to get the API key from [OpenAI] and store it locally in the `.env` file.

[OpenAI]:      https://openai.com
</details>

## 🤔How to Use the App

<details><summary><b>Show intructions</b></summary>

Once the app is launched in the browser, you will see the following:

<p align='center'>
	<img src='./img/enter-url.png', alt='Enter URL', width='650'>
</p>

You can paste a URL link that contains some text, such as an article, a blog post, or an essay. Hit Ctrl + Enter once you paste the URL.

A summary of the article will displayed in a few minutes. See example below for an [article] about when to use classes in Python.
[article]:     https://death.andgravity.com/same-functions

<p align='center'>
	<img src='./img/example-summary.png', alt='Example Summary', width='750'>
</p>

Once a summary is displayed, you can start chatting with the agent and ask questions about the document. Here are some examples:

<p align='center'>
	<img src='./img/example-qa.png', alt='Example QA', width='750'>
</p>

Note that the agent does have memory about recent questions. In the follow-up question above, I asked if there are any alternatives to it, and the agent knew what Imeant by "it" (using Python classes).
</details>

## ⚙️How It Works

<details><summary><b>Show details</b></summary>

Here's the list of tools used to develop this app:

<p align='center'>
	<img src='./img/tools-used.png', alt='Tools used', width='650'>
</p>

[Streamlit](https://streamlit.io)
[LangChain](https://docs.langchain.com)
[FAISS](https://faiss.ai)
[OpenAI](https://openai.com/product)

The text from the document is loaded and summarized using LangChain.

Then the document is divided into multiple chunks, each of which are passed to OpenAI API to create embeddings. These embeddings are stored in a local vector database using FAISS. 

When you ask a question, the question text gets converted into an embedding vector, which is then compared with all vectors that are available in the vector database. The most appropriate chunk of text is returned and passed on to OpenAI API as context for the question.

Please note that the app uses **gpt-3.5-turbo-16k** from OpenAI. You can change this, and some other settings in `config.py`.

</details>

## 💡Potential Improvements

<details><summary><b>Show details</b></summary>

Here are some improvements that can enhance the functionality or utility of this app:

1. Allow users to upload PDF document as well. Currently, the app only accepts URL links as inputs.
2. Instead of typing their questions, allow the users to ask their question by using voice (microphone).
3. Include some error handling.
</details>

### ❤️Sources

<details><summary><b>Show sources</b></summary>
[LLM Chain Documentation: Agent with Memory](https://python.langchain.com/docs/modules/memory/agent_with_memory)

[Streamlit Tutorial: Build Conversational App](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)

</details>
