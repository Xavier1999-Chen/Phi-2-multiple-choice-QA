from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
# zhipuai_api_key = "2d100eef7950cc860710345fa29c8b69.VtfECNPQNlJ205C9"
prompt = f'''
example 1:
Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
Candidate answers: (A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion 
Gold answer: A

Query:(Do not give reason.)
Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?
Candidate answers: A Planetary density will decrease. B Planetary years will become longer. C Planetary days will become shorter. D Planetary gravity will become stronger.
Gold answer:
'''

from zhipuai import ZhipuAI
client = ZhipuAI(api_key="2d100eef7950cc860710345fa29c8b69.VtfECNPQNlJ205C9") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature = 0.1,
)
print(response.choices[0].message.content[0])

# messages = [HumanMessage(content=prompt)]
# chat = ChatZhipuAI(
#     temperature=0,
#     api_key=zhipuai_api_key,
#     model="chatglm_turbo",
# )
# response = chat(messages)
# response = response.content[1:-1].replace('\\n', '\n')
# print(response)