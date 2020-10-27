import json 

fd = open('../data/exp1/all_funcs_need.json')
content = fd.read()
json_content = json.loads(content)
test_content = json_content[59876:]
fd.close()
# print(len(test_content))

res = []
for item in test_content:
    res.append(item['function'])

fd = open('../data/exp1/test.json', 'w+')
fd.write(json.dumps(json.dumps(res)))
