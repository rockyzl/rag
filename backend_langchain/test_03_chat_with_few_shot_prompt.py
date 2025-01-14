from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"a": "2+2", "output": "4"},
    {"a": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{a}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{a}"),
    ]
)

#print(final_prompt.format(input = "What's the square of a triangle?"))

llm = ChatOpenAI(temperature=0)

chain = final_prompt | llm

# Invoke the chain. 
result = chain.invoke({"a": "9+9"})

print(result.content)
