import re
import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ----- API keys -----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME") or st.secrets.get("PINECONE_INDEX_NAME", "mp2-part3-index")

# ----- Agent classes (MP2 Part-3 multi-agent pipeline) -----


class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = (
            "You are an obnoxious filter. Your job is to check if a user's query is rude, offensive, or inappropriate, "
            "or if it violates any social conduct expected in a classroom or professional environment. "
            "If the message is obnoxious, respond with 'OBNOXIOUS'. Otherwise, respond only with 'NOT OBNOXIOUS' and nothing else."
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        response_str = (response or "").strip().upper()
        if response_str == "OBNOXIOUS":
            return True
        if response_str == "NOT OBNOXIOUS":
            return False
        return False

    def check_query(self, query):
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=10,
            temperature=0,
        )
        reply = response.choices[0].message.content
        if reply is None:
            return False
        return self.extract_action(reply)


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings, embedding_dimensions=None) -> None:
        self.pinecone_index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.embedding_dimensions = embedding_dimensions
        self.prompt = None

    def query_vector_store(self, query, k=5):
        if not query or not self.embeddings:
            return []
        kwargs = {"input": query, "model": self.embeddings}
        if self.embedding_dimensions is not None:
            kwargs["dimensions"] = self.embedding_dimensions
        embedding_response = self.client.embeddings.create(**kwargs)
        query_embedding = embedding_response.data[0].embedding
        pinecone_response = self.pinecone_index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
        )
        results = []
        for match in pinecone_response.matches:
            results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            })
        return results

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response, query=None):
        if not response:
            return None
        if isinstance(response, dict):
            action = response.get("action")
            if action:
                return action
        match = re.search(r"Action\s*:\s*([^\n]+)", str(response), re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"[[(]([a-zA-Z_ ]+)[\])]", str(response))
        if match:
            return match.group(1).strip()
        return None


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.openai_client = openai_client

    def generate_response(self, query, docs, conv_history, k=5):
        if not docs:
            system_prompt = "You are a helpful assistant. Answer the user's question, but you have no relevant documents to base your answer on."
            doc_snippets = ""
        else:
            system_prompt = "You are a helpful assistant. Answer the user's question using ONLY the following context documents and your previous conversation history."
            doc_snippets = "\n".join(
                [f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(docs[:k])]
            )
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[{"role": msg["role"], "content": msg["content"]} for msg in conv_history],
                    {"role": "user", "content": f"{query}\n\nContext:\n{doc_snippets}"},
                ],
                max_tokens=250,
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Sorry, I could not process your request because of a system error. ({e})"


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.openai_client = openai_client

    def get_relevance(self, conversation) -> str:
        prompt = (
            "You judge if the CONTEXT DOCUMENTS below are useful for answering the USER's question.\n"
            "If the context documents contain information that can answer the user's question, reply with exactly: Relevant\n"
            "If the context is unrelated or cannot answer the question, reply with exactly: Irrelevant\n"
            "=== USER QUESTION ===\n"
        )
        for msg in conversation:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += "\nYour one-word answer (Relevant or Irrelevant):"
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You answer with one word only: Relevant (if the context documents help answer the user's question) or Irrelevant (if they do not).",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=15,
                temperature=0,
            )
            response = (completion.choices[0].message.content or "").strip()
            r = response.lower()
            if "irrelevant" in r:
                return "Irrelevant"
            if "relevant" in r:
                return "Relevant"
            return "Irrelevant"
        except Exception as e:
            return f"Error judging relevance: {e}"


class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name
        self.obnoxious_agent = None
        self.relevant_documents_agent = None
        self.retriever_agent = None
        self.chat_agent = None

    def setup_sub_agents(self):
        client = OpenAI(api_key=self.openai_key)
        pinecone_index = Pinecone(api_key=self.pinecone_key).Index(self.pinecone_index_name)
        self.obnoxious_agent = Obnoxious_Agent(client)
        self.relevant_documents_agent = Relevant_Documents_Agent(client)
        self.retriever_agent = Query_Agent(
            pinecone_index, client, "text-embedding-3-small", embedding_dimensions=512
        )
        self.chat_agent = Answering_Agent(client)

    def process_turn(self, user_input: str, conversation: list) -> tuple[str, str]:
        if (
            self.obnoxious_agent is None
            or self.relevant_documents_agent is None
            or self.retriever_agent is None
            or self.chat_agent is None
        ):
            self.setup_sub_agents()
        if self.obnoxious_agent.check_query(user_input):
            return "I can't respond to that.", "Obnoxious_Agent"
        top_docs = self.retriever_agent.query_vector_store(user_input)
        meta = lambda d: d.get("metadata") or {}
        docs = [{"content": meta(d).get("text", meta(d).get("content", ""))} for d in top_docs]
        context_text = "Context documents (retrieved for the user's question):\n\n" + "\n".join(d["content"] for d in docs)
        conv_relevance = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": context_text},
        ]
        if not top_docs or self.relevant_documents_agent.get_relevance(conv_relevance) != "Relevant":
            return "This question is outside the scope I can help with.", "Relevant_Documents_Agent"
        bot_response = self.chat_agent.generate_response(user_input, docs, conversation)
        return bot_response, "Chat_Agent"


# ----- Streamlit UI -----

st.title("Mini Project 2: Streamlit Chatbot (Part-3 Multi-Agent)")


@st.cache_resource
def get_head_agent():
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        return None
    agent = Head_Agent(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME)
    agent.setup_sub_agents()
    return agent


def get_conversation_for_agent():
    out = []
    for msg in st.session_state.get("messages", []):
        out.append({"role": msg["role"], "content": msg["content"]})
    return out


if "messages" not in st.session_state:
    st.session_state["messages"] = []

head_agent = get_head_agent()
if head_agent is None:
    st.warning(
        "Set **OPENAI_API_KEY** and **PINECONE_API_KEY** (and optionally **PINECONE_INDEX_NAME**) "
        "in `.env` or Streamlit secrets to use the multi-agent pipeline. "
        "Otherwise the app cannot run."
    )
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("agent_path"):
            st.caption(f"Agent: {message['agent_path']}")

if prompt := st.chat_input("What would you like to chat about?"):
    conversation = get_conversation_for_agent()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Running multi-agent pipeline…"):
        bot_response, agent_path = head_agent.process_turn(prompt, conversation)
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        st.caption(f"Agent: {agent_path}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "agent_path": agent_path,
    })
    st.rerun()
