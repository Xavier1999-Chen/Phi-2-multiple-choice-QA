# Phi-2-multiple-choice-QA

If you want to try the phi-2 + glm-4, you need to open eval_fewshot.py, uncomment the following code, and paste your own API key:

```
        with torch.no_grad():
            # task 6
            outputs = model(**encoding)
#             prompt = f'''
#             example 1:
#             Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
#             Candidate answers: (A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion 
#             Gold answer: A

#             Query:(Do not give reason. 只需要给出答案选项，不要有任何多余内容)
#             Question: {question}
#             Candidate answers: {candidate_answers}
#             Gold answer:
#             '''

#             from zhipuai import ZhipuAI
#             client = ZhipuAI(api_key="") # 填写您自己的APIKey
#             response = client.chat.completions.create(
#                 model="glm-4",  # 填写需要调用的模型名称
#                 messages=[
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature = 0.1,
#             )
#             # print(response.choices[0].message.content[0])
            log_likelihood = outputs.loss * -1
            # label_temp = problems[i]["label"]
            # print(label_temp, response.choices[0].message.content[0])
            # if label_temp == response.choices[0].message.content[0]:
            #     log_likelihood += 1
            #     print(f"Best Answer: {label_temp}!")
```