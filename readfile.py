import json
filepath='D:\\计算机\\代码\\finetune\\translation2019zh\\translation2019zh_valid.json'
'''with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            import json'''

# 由于文件中有多行，直接读取会出现错误，因此一行一行读取
file = open("D:\\计算机\\代码\\finetune\\translation2019zh\\translation2019zh_valid.json", 'r', encoding='utf-8')
papers = []
for line in file.readlines():
    dic = json.loads(line)
    papers.append(dic)


print(papers)

