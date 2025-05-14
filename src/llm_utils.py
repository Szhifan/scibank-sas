from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import OpenAI
import os
from nltk.tokenize import word_tokenize
import nltk
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
def get_scibank_response(question, student_answer, reference_answer,llm):
    prompt = get_scibank_prompt(
        question=question,
        student_answer=student_answer,
        reference_answer=reference_answer
    )
    response = llm.invoke(prompt)
    return response
if __name__ == "__main__": 
    from data_utils import SB_Dataset
    import json
    import tqdm 
    dataset = SB_Dataset()
    split = dataset.data_dict["val"]
    output_file = "data/responses_val.jsonl"
    split = [item for item in split]
    try:
        with open(output_file, "r") as f:
            start = len(f.readlines())
    except FileNotFoundError:
        start = 0  # If the file doesn't exist, start from the beginning

    with open(output_file, "a") as outfile:  # Open in append mode to continue from the latest
        for idx, item in enumerate(tqdm.tqdm(split[start:], initial=start, total=len(split))):
            try:
                question = item["question"]
                student_answer = item["student_answer"]
                reference_answer = item["reference_answer"]
                response = get_scibank_response(question, student_answer, reference_answer, llm)
                id = item["id"]
                json.dump({id:{"gpt3.5_response": response}}, outfile)
                outfile.write("\n")
            except Exception as e:
                print(f"An error occurred at index {start + idx}: {e}")
                print("Saving progress and exiting...")
                outfile.flush()
                break