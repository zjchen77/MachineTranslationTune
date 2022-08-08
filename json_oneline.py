import json

with open('D:\\计算机\\代码\\finetune\\translation2019zh\\TM-train.json', encoding='utf-8') as handle:
    data = json.load(handle)

text = json.dumps(data, separators=(',', ':'))
print(text)