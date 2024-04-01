from transformers import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from setproctitle import setproctitle
from sklearn.model_selection import train_test_split
from itertools import combinations

import torch
import torch.nn as nn
import random
import time
import datetime
import numpy as np
import pandas as pd
import os, re
import argparse

''' 데이터 클리닝 '''
def clean_data(script, data_type="dir"):
    if data_type == "dir":
        with open(script, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            preproc_lines = []
            for line in lines:
                if line.lstrip().startswith('#'):
                    continue
                if '//' in line:
                    line = line[:line.index('//')]
                line = line.replace('    ', '\t')
                if line.strip() == '':
                    continue 
                preproc_lines.append(line)

    elif data_type == "file":
        lines = script.split('\n')
        preproc_lines = []
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            if '//' in line:
                line = line[:line.index('//')]
            line = line.replace('    ', '\t')
            if line.strip() == '':
                continue 
            preproc_lines.append(line)

    preprocessed_script = '\n'.join(preproc_lines)
    preprocessed_script = re.sub('/\*.*?\*/', '', preprocessed_script, flags=re.DOTALL)
    preprocessed_script = re.sub(r'\s{2,}', ' ', preprocessed_script)

    return preprocessed_script


def data_preprocess(args):
    # 데이콘이 제공해준 학습 코드 데이터 데이터프레임 만들기
    code_folder = "train_code"  # 데이콘이 제공해준 학습 데이터 파일의 경로
    problem_folders = os.listdir(code_folder)
    preproc_scripts = []
    problem_nums = []

    for problem_folder in tqdm(problem_folders):   
        scripts = os.listdir(os.path.join(code_folder, problem_folder))
        problem_num = scripts[0].split('_')[0]
        for script in scripts:
            script_file = os.path.join(code_folder, problem_folder, script)
            preprocessed_script = clean_data(script_file, data_type="dir") 
            preproc_scripts.append(preprocessed_script)
        problem_nums.extend([problem_num] * len(scripts))
    train_df = pd.DataFrame(data={'code': preproc_scripts, 'problem_num': problem_nums})
    train_df = train_df.dropna(axis=0) #추가

    train_df.to_csv("./data/train_cleaned.csv", index=False)

    # 데이콘이 제공해준 테스트 코드 데이터 데이터프레임 만들기
    test_df = pd.read_csv("test.csv")
    code1 = test_df['code1'].values
    code2 = test_df['code2'].values
    processed_code1 = []
    processed_code2 = []
    for i in tqdm(range(len(code1))):
        processed_c1 = clean_data(code1[i], data_type="file") #수정
        processed_c2 = clean_data(code2[i], data_type="file") #수정

        processed_code1.append(processed_c1)
        processed_code2.append(processed_c2)
    processed_test = pd.DataFrame(list(zip(processed_code1, processed_code2)), columns=["code1", "code2"])

    processed_test.to_csv("./data/new_dataset_0604/processed_test.csv", index=False)
    
    

    df = pd.read_csv("./data/train_cleaned.csv")

    # train과 validation data set 분리
    train_df, valid_df, train_label, valid_label = train_test_split(
            df,
            df['problem_num'],
            random_state=42,
            test_size=0.1,
            stratify=df['problem_num']
        )

    train_df = train_df.reset_index(drop=True) # Reindexing
    valid_df = valid_df.reset_index(drop=True)

    codes = train_df['code'].to_list() # code 컬럼을 list로 변환 - codes는 code가 쭉 나열된 형태임
    problems = train_df['problem_num'].unique().tolist() # 문제 번호를 중복을 제외하고 list로 변환
    problems.sort()

    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(problems):
        # 각각의 문제에 대한 code를 골라 정답 코드로 저장, 아닌 문제는 other_codes로 저장
        # 이때 train_df에는 problem_num이 정렬된 상태가 아니기 때문에 index가 다를 수 있음
        solution_codes = train_df[train_df['problem_num'] == problem]['code'].to_list()
        other_codes = train_df[train_df['problem_num'] != problem]['code'].to_list()
        
        # positive_pairs 1800개 (총 500 * 1800 = 900,000개) 추출
        # negative_pairs 1800개 (총 500 * 1800 = 900,000개) 추출
        positive_pairs = list(combinations(solution_codes,2))
        random.shuffle(positive_pairs)
        positive_pairs = positive_pairs[:1800] 
        random.shuffle(other_codes)
        other_codes = other_codes[:1800] 
        
        negative_pairs = []
        for pos_codes, others in zip(positive_pairs, other_codes):
            negative_pairs.append((pos_codes[0], others))
        
        total_positive_pairs.extend(positive_pairs)
        total_negative_pairs.extend(negative_pairs)

    # total_positive_pairs와 negative_pairs의 정답 코드를 묶어 code1로 지정
    # total_positive_pairs와 negative_pairs의 비교 대상 코드를 묶어 code2로 지정
    # 해당 코드에 맞는 label 설정
    code1 = [code[0] for code in total_positive_pairs] + [code[0] for code in total_negative_pairs]
    code2 = [code[1] for code in total_positive_pairs] + [code[1] for code in total_negative_pairs]
    label = [1]*len(total_positive_pairs) + [0]*len(total_negative_pairs)

    # DataFrame으로 선언
    train_data = pd.DataFrame(data={'code1':code1, 'code2':code2, 'similar':label})
    train_data = train_data.sample(frac=1).reset_index(drop=True) # frac: 추출할 표본 비율

    train_data.to_csv("./data/" + "new_dataset_0607/dacon_train_random.csv", index=False)

    codes = valid_df['code'].to_list() # code 컬럼을 list로 변환 - codes는 code가 쭉 나열된 형태임
    problems = valid_df['problem_num'].unique().tolist() # 문제 번호를 중복을 제외하고 list로 변환
    problems.sort()

    total_positive_pairs = []
    total_negative_pairs = []
    
    for problem in tqdm(problems):
        # 각각의 문제에 대한 code를 골라 정답 코드로 저장, 아닌 문제는 other_codes로 저장
        # 이때 train_df에는 problem_num이 정렬된 상태가 아니기 때문에 index가 다를 수 있음
        solution_codes = valid_df[valid_df['problem_num'] == problem]['code'].to_list()
        other_codes = valid_df[valid_df['problem_num'] != problem]['code'].to_list()
        
        # positive_pairs 100개 (총 300 * 100 = 30,000개) 추출
        # negative_pairs 100개 (총 300 * 100 = 30,000개) 추출
        positive_pairs = list(combinations(solution_codes,2))
        random.shuffle(positive_pairs)
        positive_pairs = positive_pairs[:100]
        random.shuffle(other_codes)
        other_codes = other_codes[:100]
        
        negative_pairs = []
        for pos_codes, others in zip(positive_pairs, other_codes):
            negative_pairs.append((pos_codes[0], others))
        
        total_positive_pairs.extend(positive_pairs)
        total_negative_pairs.extend(negative_pairs)


    # total_positive_pairs와 negative_pairs의 정답 코드를 묶어 code1로 지정
    # total_positive_pairs와 negative_pairs의 비교 대상 코드를 묶어 code2로 지정
    # 해당 코드에 맞는 label 설정
    code1 = [code[0] for code in total_positive_pairs] + [code[0] for code in total_negative_pairs]
    code2 = [code[1] for code in total_positive_pairs] + [code[1] for code in total_negative_pairs]
    label = [1]*len(total_positive_pairs) + [0]*len(total_negative_pairs)

    # DataFrame으로 선언
    valid_data = pd.DataFrame(data={'code1':code1, 'code2':code2, 'similar':label})
    valid_data = valid_data.sample(frac=1).reset_index(drop=True) # frac: 추출할 표본 비율

    valid_data.to_csv("./data/" + "new_dataset_0607/dacon_valid_random.csv", index=False)

   

def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)



def train_model(args):

    set_seed(args)
    setproctitle(args.process_name)

    train_data = pd.read_csv("./data/" + "new_dataset_0607/dacon_train_random.csv")
    valid_data = pd.read_csv("./data/" + "new_dataset_0607/dacon_valid_random.csv")

    # training
    c1 = train_data['code1'].values
    c2 = train_data['code2'].values
    similar = train_data['similar'].values

    N = train_data.shape[0]
    MAX_LEN = 512

    input_ids = np.zeros((N, MAX_LEN), dtype=int)
    attention_masks = np.zeros((N, MAX_LEN), dtype=int)
    labels = np.zeros((N), dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

    for i in tqdm(range(N), position=0, leave=True):
        try:
            cur_c1 = str(c1[i])
            cur_c2 = str(c2[i])
            encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                      truncation=True)
            input_ids[i,] = encoded_input['input_ids']
            attention_masks[i,] = encoded_input['attention_mask']
            labels[i] = similar[i]
        except Exception as e:
            print(e)
            pass


    # validating
    c1 = valid_data['code1'].values
    c2 = valid_data['code2'].values
    similar = valid_data['similar'].values

    N = valid_data.shape[0]

    MAX_LEN = 512

    valid_input_ids = np.zeros((N, MAX_LEN), dtype=int)
    valid_attention_masks = np.zeros((N, MAX_LEN), dtype=int)
    valid_labels = np.zeros((N), dtype=int)

    for i in tqdm(range(N), position=0, leave=True):
        try:
            cur_c1 = str(c1[i])
            cur_c2 = str(c2[i])
            encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                      truncation=True)
            valid_input_ids[i,] = encoded_input['input_ids']
            valid_attention_masks[i,] = encoded_input['attention_mask']
            valid_labels[i] = similar[i]
        except Exception as e:
            print(e)
            pass

    if os.path.exists(args.dir_path):
        os.makedirs(args.dir_path, exist_ok=True)

    print("\n\nMake tensor\n\n")
    input_ids = torch.tensor(input_ids, dtype=int)
    attention_masks = torch.tensor(attention_masks, dtype=int)
    labels = torch.tensor(labels, dtype=int)

    valid_input_ids = torch.tensor(valid_input_ids, dtype=int)
    valid_attention_masks = torch.tensor(valid_attention_masks, dtype=int)
    valid_labels = torch.tensor(valid_labels, dtype=int)


    if args.save_tensor == True:
        torch.save(input_ids, "./data/" + args.dir_path + "/" + args.model_name + '_mixed_train_input_ids.pt')
        torch.save(attention_masks, "./data/" + args.dir_path + "/" + args.model_name + '_mixed_train_attention_masks.pt')
        torch.save(labels, "./data/" + args.dir_path + "/" + args.model_name + '_mixed_train_labels.pt')

        torch.save(valid_input_ids, "./data/" + args.dir_path + "/" + args.model_name + "_mixed_valid_input_ids.pt")
        torch.save(valid_attention_masks, "./data/" + args.dir_path + "/" + args.model_name + "mixed_valid_attention_masks.pt")
        torch.save(valid_labels, "./data/" + args.dir_path + "/" + args.model_name + "mixed_valid_labels.pt")


    # Setup training
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    train_data = TensorDataset(input_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    validation_data = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-5)  

    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device("cuda")
    loss_f = nn.CrossEntropyLoss()

    # Train
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    model.zero_grad()
    for i in range(args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(i + 1, args.epochs))
        print('Training...')
        t0 = time.time()
        train_loss, train_accuracy = 0, 0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration", smoothing=0.05):
            if step % 10000 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                print('  current average loss = {}'.format(
                    train_loss / step))  # bot.sendMessage(chat_id=chat_id, text = '  current average loss = {}'.format(train_loss / step))

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.detach().cpu().numpy()
            train_accuracy += flat_accuracy(logits, label_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_accuracy = train_accuracy / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        print("  Average training loss: {0:.8f}".format(avg_train_loss))
        print("  Average training accuracy: {0:.8f}".format(avg_train_accuracy))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Validating...")
        t0 = time.time()
        model.eval()
        val_loss, val_accuracy = 0, 0
        for step, batch in tqdm(enumerate(validation_dataloader), desc="Iteration", smoothing=0.05):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu()
            label_ids = b_labels.detach().cpu()
            val_loss += loss_f(logits, label_ids)

            logits = logits.numpy()
            label_ids = label_ids.numpy()
            val_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = val_accuracy / len(validation_dataloader)
        avg_val_loss = val_loss / len(validation_dataloader)
        val_accuracies.append(avg_val_accuracy)
        val_losses.append(avg_val_loss)
        print("  Average validation loss: {0:.8f}".format(avg_val_loss))
        print("  Average validation accuracy: {0:.8f}".format(avg_val_accuracy))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # if np.min(val_losses) == val_losses[-1]:
        print("saving current best checkpoint")
        torch.save(model.state_dict(), "./data/" + args.dir_path + "/" + str(i + 1) + "_mixed_" + args.model_name + "_random.pt")


def inference_model(args):
    test_data = pd.read_csv("./data/new_dataset_0604/processed_test.csv")

    c1 = test_data['code1'].values
    c2 = test_data['code2'].values

    N = test_data.shape[0]
    MAX_LEN = 512

    test_input_ids = np.zeros((N, MAX_LEN), dtype=int)
    test_attention_masks = np.zeros((N, MAX_LEN), dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    tokenizer.truncation_side = "left"

    for i in tqdm(range(N), position=0, leave=True):
        try:
            cur_c1 = str(c1[i])
            cur_c2 = str(c2[i])
            encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                      truncation=True)
            test_input_ids[i,] = encoded_input['input_ids']
            test_attention_masks[i,] = encoded_input['attention_mask']

        except Exception as e:
            print(e)
            pass

    test_input_ids = torch.tensor(test_input_ids, dtype=int)
    test_attention_masks = torch.tensor(test_attention_masks, dtype=int)

    if args.save_tensor == True:
        torch.save(test_input_ids, "./data/" + args.dir_path + "/" + "test_input_ids_0605.pt")
        torch.save(test_attention_masks, "./data/" + args.dir_path + "/" + "test_attention_masks_0605.pt")

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)
    PATH = "./data/" + args.dir_path + "/" + "2_mixed_" + args.model_name + "_random.pt"

    model.load_state_dict(torch.load(PATH))
    model.cuda()

    test_tensor = TensorDataset(test_input_ids, test_attention_masks)
    test_sampler = SequentialSampler(test_tensor)
    test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=args.test_batch_size)

    submission = pd.read_csv('sample_submission.csv')
    device = torch.device("cuda")

    preds = np.array([])
    for step, batch in tqdm(enumerate(test_dataloader), desc="Iteration", smoothing=0.05):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu()
        _pred = logits.numpy()
        pred = np.argmax(_pred, axis=1).flatten()
        preds = np.append(preds, pred)

    submission['similar'] = preds
    submission.to_csv('./data/submission_' + args.model_name + '_random.csv', index=False)


def model_ensemble():
    submission = pd.read_csv('sample_submission.csv')

    submission_1 = pd.read_csv('./data/submission_graphcodebert_random.csv')
    submission_2 = pd.read_csv('./data/submission_codebert-cpp_random.csv')

    sub_1 = submission_1['similar']
    sub_2 = submission_2['similar']

    ensemble_preds = (sub_1 + sub_2) / 2

    preds = np.where(ensemble_preds >= 0.5, 1, 0)

    submission['similar'] = preds

    submission.to_csv('./data/submission_ensemble_random_graph_cobertcpp각180만_0.5=1.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set arguments.")

    parser.add_argument("--seed", default="42", type=int, help="Random seed for initialization")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--eps", default=1e-5, type=float, help="The initial eps.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=None, help="batch_size")
    parser.add_argument("--test_batch_size", type=int, default=None, help="test_batch_size")

    parser.add_argument("--no_cuda", default=False, type=bool, help="Say True if you don't want to use cuda.")
    parser.add_argument("--ensemble", default=False, type=bool, help="Ensemble.")
    parser.add_argument("--save_tensor", default=True, type=str, help="Save tensor.")
    parser.add_argument("--mode", default="train", type=str, help="When you train the model.")
    parser.add_argument("--dir_path", default="graphcodebert", type=str, help="Save model path.")
    parser.add_argument("--model_name", default="graphcodebert", type=str, help="Model name.")
    parser.add_argument("--process_name", default="code_similarity", type=str, help="process_name.")
    parser.add_argument("--checkpoint_path", default="microsoft/graphcodebert-base", type=str, help="Pre-trained Language Model.")

    args = parser.parse_args()

    if args.mode == "train":
        data_preprocess(args)
        train_model(args)
    else:
        inference_model(args)

    if args.ensemble == True:
        model_ensemble()

    # CUDA_VISIBLE_DEVICES=0 python code_submission.py --seed 42 --learning_rate 2e-5 --eps 1e-5 --epochs 3 --batch_size 32 --test_batch_size 1048 --save_tensor True --mode train --dir_path graphcodebert --model_name graphcodebert --process_name code_similarity --checkpoint_path microsoft/graphcodebert-base