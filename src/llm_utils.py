from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAI
import os
from nltk.tokenize import word_tokenize
import nltk
from src.data_prep import SB_Dataset
path_cred = "credentials_openai.txt"
with open(path_cred, "r") as file:
    lines = file.readlines()
    api_key = lines[0].strip()

os.environ["OPENAI_API_KEY"] = api_key

llm = OpenAI(
    name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=400,
)


def get_scibank_prompt(question, student_answer, reference_answer):
    prompt = ChatPromptTemplate.from_messages([
  ("system", "Assess student answers to short-answer questions based on a given reference answer."),
  ("human", "Question: {question}\n\nStudent Answer: {student_answer}\n\nReference Answer: {reference_answer}\n\nPlease assess the student’s answer.")
]

    )
 
    return prompt.format_messages(
        question=question,
        student_answer=student_answer,
        reference_answer=reference_answer
    )

if __name__ == "__main__": 
    pass 