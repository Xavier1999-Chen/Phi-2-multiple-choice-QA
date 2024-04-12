import argparse
import pprint
import os
import copy
from str2bool import str2bool
from typing import Dict, Sequence
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

IGNORE_INDEX = -100

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--device_id', type=str, default="")
parser.add_argument('--model', type=str, default='', help="")
parser.add_argument('--embedder', type=str, default="")
parser.add_argument('--output_path', type=str, help="")
parser.add_argument('--start_index', type=int, default=0, help="")
parser.add_argument('--end_index', type=int, default=164, help="")
parser.add_argument('--N', type=int, default=8, help="")
parser.add_argument('--max_len', type=int, default=512, help="")
parser.add_argument('--overwrite', type=str2bool, default=False, help="")
parser.add_argument('--prompt_type', type=str, default="v1.0", help="")
parser.add_argument('--top_k', type=str2bool, default=False, help="")
parser.add_argument('--top_k_reverse', type=str2bool, default=False, help="")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)


from tqdm import tqdm
import torch
import json

import transformers
from modeling_phi import PhiForCausalLM
from tokenization_codegen import CodeGenTokenizer


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_arc_problems(data_path="data/ARC-Easy-test.jsonl"):
    dataset = []
    with open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            candidate_answers = " ".join([f"({label}) {text}" for text, label in zip(json_obj["choices"]["text"], json_obj["choices"]["label"])]).strip()
            for text, label in zip(json_obj["choices"]["text"], json_obj["choices"]["label"]):
                dataset.append({
                    "id": json_obj["id"],
                    "question": json_obj["question"],
                    "candidate_answers": candidate_answers,
                    "answer": text,
                    "label": label,
                    "answerKey": json_obj["answerKey"],
                })
    return dataset


def load_all_demonstrations(train_path="data/ARC-Challenge-train.jsonl"):
    demonstrations = []
    with open(train_path, encoding="utf-8") as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            demonstrations.append((json_obj["question"], json_obj["choices"]["text"], json_obj["choices"]["label"], json_obj["answerKey"]))
    print(f"load {len(demonstrations)} demonstrations from {train_path}")
    return demonstrations


def llm_embedder(llm, sentences, is_query=True):
    INSTRUCTIONS = {
        "qa": {
            "query": "Represent this query for retrieving relevant documents: ",
            "key": "Represent this document for retrieval: ",
        },
        "icl": {
            "query": "Convert this example into vector to look for useful examples: ",
            "key": "Convert this example into vector for retrieval: ",
        },
        "chat": {
            "query": "Embed this dialogue to find useful historical dialogues: ",
            "key": "Embed this historical dialogue for retrieval: ",
        },
        "lrlm": {
            "query": "Embed this text chunk for finding useful historical chunks: ",
            "key": "Embed this historical text chunk for retrieval: ",
        },
        "tool": {
            "query": "Transform this user request for fetching helpful tool descriptions: ",
            "key": "Transform this tool description for retrieval: "
        },
        "convsearch": {
            "query": "Encode this query and context for searching relevant passages: ",
            "key": "Encode this passage for retrieval: ",
        },
    }

    instruction = INSTRUCTIONS["icl"]
    if is_query:
        sentences = [instruction["query"] + s for s in sentences]
    else:
        sentences = [instruction["key"] + s for s in sentences]

    # Encode
    sentence_embeddings = llm.encode(sentences)
    return sentence_embeddings

def candidate_answers_formating(texts, labels):
    candidate_answers = " ".join([f"({label}) {text}" for text, label in zip(texts, labels)]).strip()
    return candidate_answers

# task 4
def example_formating(question, answer=None, candidate_answers=None, prompt_type="v2.0"):
    if prompt_type == "v1.0":
        if answer is not None:
            prompt = f"Question: {question}\nCandidate answers: {candidate_answers}\nGold answer: {answer}"
        else:
            prompt = f"Question: {question}\nCandidate answers: {candidate_answers}\nGold answer:"
    elif prompt_type == "v2.0":
        if answer is not None:
            prompt = f"""
            Example:
            Question: {question}
            Candidate answers: {candidate_answers}
            Gold answer: {answer}"""#"Write Your Code Here"
        else:
            prompt = f"""
            Query:(Do not give reason.)
            Question: {question}
            Candidate answers: {candidate_answers}
            Gold answer:"""#"Write Your Code Here"
    else:
        raise NotImplementedError
    return prompt

def generate_prompt(question, candidate_answers, prompt_type, N,
                    demonstrations, demonstration_embeddings, embedder,
                    top_k=False, top_k_reverse=False):

    indices = list(range(len(demonstrations)))
    if top_k: # task 5
        question_embeddings = llm_embedder(embedder, [question], True) # [1, n_dim]
        similarity =  question_embeddings @ demonstration_embeddings.T #"Write Your Code Here" @ "Write Your Code Here" # [1, n_demo]
        indices_sorted = sorted(list(range(len(demonstrations))), key=lambda x: similarity[0][x], reverse=True)
        if top_k_reverse:
            indices = indices_sorted[:N][::-1] + indices_sorted[N:]
        else:
            indices = indices_sorted

    template = ""
    for idx in indices[:N]:
        demo = demonstrations[idx]
        candidate = candidate_answers_formating(demo[1], demo[2])
        gold = demo[1][demo[2].index(demo[3])]
        template += f"\n\n{example_formating(demo[0], answer=gold, candidate_answers=candidate, prompt_type=prompt_type)}"

    template += f"\n\n{example_formating(question, candidate_answers=candidate_answers, prompt_type=prompt_type)}"

    return template.strip()


def get_model(
    base_model: str = "bigcode/starcoder",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = CodeGenTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = PhiForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=args.max_len,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=torch.stack(input_ids).to(device), labels=torch.stack(labels).to(device))



def main():

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = get_arc_problems(args.data_path)[args.start_index: args.end_index]

    num_samples = len(problems)
    tokenizer, model = get_model(base_model=args.model)
    print(f"Loaded {args.model}.")

    embedder = SentenceTransformer(args.embedder, device=device)
    print(f"loaded {args.embedder}.")

    demonstrations = load_all_demonstrations(args.data_path.replace("test", "train").replace("validation", "train"))
    demonstration_embeddings = llm_embedder(embedder, [d[0] for d in demonstrations], False) # ndarray: [n_demons, n_dim]

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        question = problems[i]["question"]
        answer = problems[i]["answer"]
        candidate_answers = problems[i]["candidate_answers"]

        source = generate_prompt(question, candidate_answers, args.prompt_type, args.N,
                                 demonstrations, demonstration_embeddings, embedder,
                                 top_k=args.top_k, top_k_reverse=args.top_k_reverse)
        if i == 0:
            print(f"prompt #{i}: {source}")

        target = " {}".format(answer)
        encoding = preprocess([source], [target], tokenizer)

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
        print("Saving results to {}".format(output_file))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": problems[i]["id"],
                "log_likelihood": log_likelihood.tolist(),
                "question": question,
                "candidate_answers": candidate_answers,
                "answer": answer,
                "label": problems[i]["label"],
                "answerKey": problems[i]["answerKey"],
            }) + "\n")




if __name__ == '__main__':
    main()
         