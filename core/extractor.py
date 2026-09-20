# Actionable items, decisions, questions — combined into a single LLM pass

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import re


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_all(transcript: str) -> dict:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript below, "
        "extract three things. Respond using EXACTLY these three section headers, "
        "each on its own line, followed by a numbered list:\n\n"
        "ACTION ITEMS:\n"
        "(numbered list — task, owner, deadline if mentioned else 'Not specified'. "
        "If none, write 'No action items found.')\n\n"
        "KEY DECISIONS:\n"
        "(numbered list. If none, write 'No key decisions found.')\n\n"
        "OPEN QUESTIONS:\n"
        "(numbered list of unresolved questions/follow-ups. If none, write "
        "'No open questions found.')"
    )
    raw = chain.invoke(transcript)
    return _split_sections(raw)


def _split_sections(raw: str) -> dict:
    sections = {"action_items": "", "key_decisions": "", "open_questions": ""}
    pattern = re.split(
        r"(ACTION ITEMS:|KEY DECISIONS:|OPEN QUESTIONS:)", raw, flags=re.IGNORECASE
    )
    current_key = None
    key_map = {
        "action items:": "action_items",
        "key decisions:": "key_decisions",
        "open questions:": "open_questions",
    }
    for part in pattern:
        stripped = part.strip().lower()
        if stripped in key_map:
            current_key = key_map[stripped]
        elif current_key:
            sections[current_key] = part.strip()

    if not any(sections.values()):
        sections["action_items"] = raw.strip()

    return sections


def extract_action_items(transcript: str) -> str:
    return extract_all(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    return extract_all(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    return extract_all(transcript)["open_questions"]
