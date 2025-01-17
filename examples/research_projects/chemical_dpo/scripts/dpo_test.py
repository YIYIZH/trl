from peft import AutoPeftModelForCausalLM
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertTokenizer, AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup, BertConfig
from datasets import load_dataset, DatasetDict
from datasets import concatenate_datasets
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import random
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset
import pandas as pd
import json
import csv
import openai

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')



test_path='./'
name='dpo316632_'
openai.api_key = '' # enter your own api key


def generate_test_data(samples):
    new_examples = {"prompt": [],"true_label":[]}
    for i in range(len(samples["prompt"])):
        new_examples["prompt"].append("Question: " + samples["prompt"][i] + "\n\nAnswer: ")
        new_examples["true_label"].append(samples["true_label"][i])
    # print(new_examples)
    return new_examples

test_dataset = load_dataset(
    "frisky11/SmartChemQA",
    split="train",
    data_files="test/dataset_smiles_632632_test.parquet"
)
# test_dataset = load_dataset(
#     "Afterglow777/chemical_dpo_dataset2",
#     split="test",
#     data_dir="data/test"
# )
original_columns = test_dataset.column_names
test_ds = test_dataset.map(
    generate_test_data,
    batched=True,
    remove_columns=original_columns
).shuffle(seed=42)


model = AutoPeftModelForCausalLM.from_pretrained(
    test_path+"dpo_results1/final_checkpoint",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
).to(device)

def encode_fn(text_list):
    all_input_ids = []    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    for text in text_list:
        input_ids = tokenizer(
                        text,  
                        add_special_tokens = False,                    
                        return_tensors = 'pt'       
                   )['input_ids']
        all_input_ids.append(input_ids)    
    return all_input_ids


tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

test_data=test_ds['prompt']
test_list = test_data
label_list = test_ds['true_label']
test_list=encode_fn(test_data)
device = torch.device('cuda')
#classifier_model=torch.load('ai_classifier_new.pt').to(device)
#accuracy=0
#output_dataset={'outputs':[],'label':[]}

#with open('results_200_gpt.csv','w',newline='') as csvfile:
    #writer = csv.writer(csvfile)
    #writer.writerow(['prompt','label'])
    
print(len(test_list))
data = {"prompt": [],
        "answer": [],
        "true_label": [],
        }
for i in range(len(test_list)):
    input_ids=test_list[i].to(device)
    outputs = model.generate(input_ids=input_ids,max_length=512)
    outputs_answer = tokenizer.decode(outputs[0],skip_special_tokens=True)
    input_text = tokenizer.decode(input_ids[0],skip_special_tokens=True)
    
    #completion = openai.ChatCompletion.create(
    #model="gpt-3.5-turbo",
    #messages=[
    #{"role": "system", "content": "You are a helpful assistant."},
    #{"role": "user", "content": input_text},
    #],
    #max_tokens=512
    #)
    #output_answer = completion.choices[0].message.content
    #print(output_answer)
    #bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    #in_classify = bert_tokenizer.encode(
                        #outputs_text,                     
                        #add_special_tokens = True,  
                        #max_length = 512,          
                        #padding = 'max_length',   
                        #return_tensors = 'pt',
                        #truncation=True       
                   #)
    #output_label = classifier_model(in_classify.to(device), token_type_ids=None, attention_mask=(in_classify>0).to(device),labels=None)[0].detach().cpu().numpy()
    #print(output_label)
    #output_label = np.argmax(output_label, axis=1).flatten()
    true_label = label_list[i]
    print(i, true_label,outputs_answer)
    prompt=input_text
    answer= outputs_answer
    data['prompt'].append(prompt)
    data['answer'].append(answer)
    data['true_label'].append(true_label)
    
    with open(test_path+name+'result.csv','a',newline='',encoding='gb18030') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([prompt,answer,true_label])

json.dump(data, open('dataset11.json', 'w'))  
