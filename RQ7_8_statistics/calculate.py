from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

import pandas as pd
import tokenize
from io import BytesIO

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
def count_tokens(code):
    tokens = tokenizer.encode(code)
    return len(tokens)
# 将函数片段分成不同区间
def split_into_intervals(lengths, interval_size):
    intervals = {'0-100':[],
                 '100-200':[],
                 '200-300':[],
                 '300-400':[],
                 '400-500':[],
                 '>500':[],}
    for i in range(len(lengths)):
        length = lengths[i]
        interval = '{}-{}'.format((length // interval_size)*interval_size,(length // interval_size)*interval_size+interval_size) if length<500 else '>500'
        intervals[interval].append(i)
    return intervals
def split_into_CWEs(CWE25,cwetocalculate):

    intervals = {}
    for i in CWE25.keys():
        intervals[i] = []
    for i in range(len(cwetocalculate)):
        tem_cwe = cwetocalculate[i]
        if tem_cwe in CWE25.keys():
            intervals[tem_cwe].append(i)
    return intervals
def getfunclen():
    models = []
    preds = []
    tem_df = pd.read_csv('./results/CodeBERT/raw_preds.csv')
    source = tem_df['processed_func'].tolist()
    target = tem_df['target'].tolist()

    for model_name in os.listdir("./results"):
        models.append(model_name)
        df=pd.read_csv('./results/{}/raw_preds.csv'.format(model_name))
        pred = df['raw_preds']
        pred = [int(i) for i in pred]
        preds.append(pred)
    # 假设你已经有了函数片段长度、缺陷标签和模型预测结果的数据
    function_lengths = [count_tokens(s) for s in source]  # 函数片段长度的列表
    defect_labels = target  # 缺陷标签的列表

    interval_size = 100
    intervals = split_into_intervals(function_lengths, interval_size)

    #F1
    df_f1 = pd.DataFrame()
    df_f1['Range'] = intervals.keys()
    df_f1['Range Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for r in intervals.keys():
        if intervals[r] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[r]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            f1 = f1_score(interval_labels, interval_predictions)
            results[models[i]].append(f1)
    for model_name in models:
        df_f1[model_name] = results[model_name]
    df_f1.to_csv('./func_len_f1.csv',index=False)

    #Accuracy
    df_accuracy = pd.DataFrame()
    df_accuracy['Range'] = intervals.keys()
    df_accuracy['Range Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for r in intervals.keys():
        if intervals[r] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[r]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            accuracy = accuracy_score(interval_labels, interval_predictions)
            results[models[i]].append(accuracy)
    for model_name in models:
        df_accuracy[model_name] = results[model_name]
    df_accuracy.to_csv('./func_len_accuracy.csv',index=False)

    #precision
    df_precision = pd.DataFrame()
    df_precision['Range'] = intervals.keys()
    df_precision['Range Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for r in intervals.keys():
        if intervals[r] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[r]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            precision = precision_score(interval_labels, interval_predictions)
            results[models[i]].append(precision)
    for model_name in models:
        df_precision[model_name] = results[model_name]
    df_precision.to_csv('./func_len_precision.csv',index=False)

    #recall
    df_recall = pd.DataFrame()
    df_recall['Range'] = intervals.keys()
    df_recall['Range Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for r in intervals.keys():
        if intervals[r] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[r]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            recall = recall_score(interval_labels, interval_predictions)
            results[models[i]].append(recall)
    for model_name in models:
        df_recall[model_name] = results[model_name]
    df_recall.to_csv('./func_len_recall.csv',index=False)
def getcwe():
    CWE25 = {
        'CWE-787':'Out-of-bounds Write',
        'CWE-79':'''Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')''',
        'CWE-89':'''Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')''',
        'CWE-416':'Use After Free',
        'CWE-78':'''Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')''',
        'CWE-20':'''Improper Input Validation''',
        'CWE-125':'''Out-of-bounds Read''',
        'CWE-22':'''Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')''',
        'CWE-352':'''Cross-Site Request Forgery (CSRF)''',
        'CWE-434':'''Unrestricted Upload of File with Dangerous Type''',
        'CWE-862':'''Missing Authorization''',
        'CWE-476':'''NULL Pointer Dereference''',
        'CWE-287':'''Improper Authentication''',
        'CWE-190':'''Integer Overflow or Wraparound''',
        'CWE-502':'''Deserialization of Untrusted Data''',
        'CWE-77':'''Improper Neutralization of Special Elements used in a Command ('Command Injection')''',
        'CWE-119':'''Improper Restriction of Operations within the Bounds of a Memory Buffer''',
        'CWE-798':'''Use of Hard-coded Credentials''',
        'CWE-918':'''Server-Side Request Forgery (SSRF)''',
        'CWE-306':'''Missing Authentication for Critical Function''',
        'CWE-362':'''Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')''',
        'CWE-269':'''Improper Privilege Management''',
        'CWE-94':'''Improper Control of Generation of Code ('Code Injection')''',
        'CWE-863':'''Incorrect Authorization''',
        'CWE-276':'''Incorrect Default Permissions'''
    }
    models = []
    preds = []
    tem_df = pd.read_csv('./results/CodeBERT/raw_preds.csv')
    target = tem_df['target'].tolist()
    cwes = tem_df['CWE ID']
    for model_name in os.listdir("./results"):
        models.append(model_name)
        df=pd.read_csv('./results/{}/raw_preds.csv'.format(model_name))
        pred = df['raw_preds']
        pred = [int(i) for i in pred]
        preds.append(pred)
    # 假设你已经有了函数片段长度、缺陷标签和模型预测结果的数据
    defect_labels = target  # 缺陷标签的列表

    intervals = split_into_CWEs(CWE25,cwes)

    #F1
    df_f1 = pd.DataFrame()
    df_f1['CWE ID'] = CWE25.keys()
    df_f1['CWE Name'] = CWE25.values()
    df_f1['CWE Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for cweid in CWE25.keys():
        if intervals[cweid] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[cweid]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            f1 = f1_score(interval_labels, interval_predictions)
            results[models[i]].append(f1)
    for model_name in models:
        df_f1[model_name] = results[model_name]
    df_f1.to_csv('./CWE_f1.csv',index=False)

    #Accuracy
    df_accu = pd.DataFrame()
    df_accu['CWE ID'] = CWE25.keys()
    df_accu['CWE Name'] = CWE25.values()
    df_accu['CWE Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for cweid in CWE25.keys():
        if intervals[cweid] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[cweid]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            accu = accuracy_score(interval_labels, interval_predictions)
            results[models[i]].append(accu)
    for model_name in models:
        df_accu[model_name] = results[model_name]
    df_accu.to_csv('./CWE_accu.csv',index=False)

    #Precision
    df_pre = pd.DataFrame()
    df_pre['CWE ID'] = CWE25.keys()
    df_pre['CWE Name'] = CWE25.values()
    df_pre['CWE Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for cweid in CWE25.keys():
        if intervals[cweid] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[cweid]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            pre = precision_score(interval_labels, interval_predictions)
            results[models[i]].append(pre)
    for model_name in models:
        df_pre[model_name] = results[model_name]
    df_pre.to_csv('./CWE_pre.csv',index=False)

    #recall
    df_recall = pd.DataFrame()
    df_recall['CWE ID'] = CWE25.keys()
    df_recall['CWE Name'] = CWE25.values()
    df_recall['CWE Number'] = [len(i) for i in intervals.values()]
    # 初始化结果存储字典
    results = {
        model : [] for model in models
    }
    for cweid in CWE25.keys():
        if intervals[cweid] ==[]:
            for i in range(len(models)):
                results[models[i]].append('-/-')
            continue
        for i in range(len(models)):
            predictions = preds[i]
            interval_indices = intervals[cweid]
            interval_predictions = [predictions[i] for i in interval_indices]
            interval_labels = [defect_labels[i] for i in interval_indices]
            recall = recall_score(interval_labels, interval_predictions)
            results[models[i]].append(recall)
    for model_name in models:
        df_recall[model_name] = results[model_name]
    df_recall.to_csv('./CWE_recall.csv',index=False)
if __name__ == '__main__':
    getfunclen()
    getcwe()