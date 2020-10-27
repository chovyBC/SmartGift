import json
import sys
import re
import datetime
from string import digits
from gensim import corpora, models, similarities
from gensim.models import Doc2Vec
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import nltk.tokenize as tk
from nltk.tokenize import word_tokenize


TaggededDocument = models.doc2vec.TaggedDocument

def process_bytes(s):
    s = json.loads(json.dumps(s))
    if type(s) is dict:
        data = s['data']
        hexstr = "0x"
        for idx in range(len(data)):
            ele = "{:02x}".format(data[idx])
            hexstr = hexstr + ele
        return hexstr
    else:
        hexstrings = []
        for item in s:
            hexstrings.append(process_bytes(item))
        return hexstrings

# give a file with the contract abi, extract all the function information
# from the abi file
def read_targets_from_abi(filepath):
    funcs_in_abi = []
    with open(filepath) as fp:
        abi = fp.read()
        abi_json = json.loads(json.loads(abi))
        for single_func in abi_json:
            if single_func["type"] == "function":
                fun_sig = {"name" : single_func["name"], "inputs" : single_func["inputs"]}
                funcs_in_abi.append(json.dumps(fun_sig))
    return funcs_in_abi

# read the database functions from the json file
def read_funcdef_from_file(filename):
    func_list = []
    input_list = []
    with open(filename) as fp:
        file = fp.read()
        json_file_list = json.loads(file)
        for i in range(len(json_file_list)):
            func_list.append(json_file_list[i]['function'])
            input_list.append(json_file_list[i]['inputs'])

    #res_list = process_functions(func_list)
    return func_list, input_list

# the input is from the contractFuzzer, randomly choosed function definition
def process_input(input):
    input_type_list = []
    test_str = ''
    json_input = json.loads(input)
    test_str += process_name(json_input["name"]) + ' '
    for item in json_input["inputs"]:
        test_str += item["type"] + ' '
        input_type_list.append(item["type"])
    return test_str, input_type_list

# process the function definion to the string format
# the format like: functionName param1 type1 param2 type2 ....
def process_functions(funclist):
    func_def_list = []
    # print(funclist)
    # first_funclist, first_inputlist,second_funclist, second_= trim_func(funclist, input_type_list, input_list, k)
    for i in range(len(funclist)):
        # cur_func = json.loads(funclist[i])
        func_str = ''
        func_str += process_name(funclist[i]["method"]) + ' '
        for j in range(len(funclist[i]["types"])):
            func_str += funclist[i]["types"][j] + ' '
        func_def_list.append(func_str)
    return func_def_list

# choose the functions that exist the type info
def trim_func(func_list, input_type_list, input_list, embedding):
    level = 'first_level'
    first_level = []
    first_level_input = []
    first_level_embedding = []
    second_level = []
    second_level_input = []
    second_level_embedding = []
    third_level = []
    third_level_input = []
    third_level_embedding = []
    remove_digits = str.maketrans('', '', digits)

    # process the first level functions
    # every type must exist
    # first level means totally the same, e.g. int256 --> int 256
    for i in range(len(func_list)):
        isFirst = False
        isSecond = False
        type_json = func_list[i]["types"]
        have_set = []
        types_are_all_exist = 0
        for input_type in input_type_list:
            # input_type = str(input_type).split('[')[0].translate(remove_digits)
            input_type = str(input_type).split('[')[0]
            for j in range(len(type_json)):
                # func_type = str(type_json[j]).split('[')[0].translate(remove_digits)
                func_type = str(type_json[j]).split('[')[0]
                if func_type == input_type and (j not in have_set):
                    types_are_all_exist += 1
                    have_set.append(j)
                    break
        if types_are_all_exist == len(input_type_list):
            isFirst = True
            first_level.append(func_list[i])
            first_level_input.append(input_list[i])
            first_level_embedding.append(embedding[i])

        if not isFirst:
            type_json = func_list[i]["types"]
            have_set = []
            types_are_all_exist = 0
            for input_type in input_type_list:
                input_type = str(input_type).split('[')[0]
                if len(re.findall("\d+", str(input_type).split('[')[0])) > 0:
                    input_type_num = int(re.findall("\d+", str(input_type).split('[')[0])[0])
                else:
                    input_type_num = 256
                for j in range(len(type_json)):
                    func_type = str(type_json[j]).split('[')[0]
                    if len(re.findall("\d+", str(type_json[j]).split('[')[0])) > 0:
                        func_type_num = int(re.findall("\d+", str(type_json[j]).split('[')[0])[0])
                    else:
                        func_type_num = 256
                    if func_type == input_type and (j not in have_set) and input_type_num >= func_type_num:
                        types_are_all_exist += 1
                        have_set.append(j)
                        break
                    # not very restrict
                    elif (("int" in func_type and "int" in input_type) or ("byte" in func_type and "byte" in input_type)) and input_type_num >= func_type_num and (j not in have_set):
                        types_are_all_exist += 1
                        have_set.append(j)
                        break
            if types_are_all_exist == len(input_type_list):
                isSecond = True
                second_level.append(func_list[i])
                second_level_input.append(input_list[i])
                second_level_embedding.append(embedding[i])
            
            if not isFirst and not isSecond:
                third_level.append(func_list[i])
                third_level_input.append(input_list[i])
                third_level_embedding.append(embedding[i])
  
    return first_level,first_level_input,first_level_embedding, second_level, second_level_input, second_level_embedding, third_level, third_level_input, third_level_embedding

# process the function name, such as aaaBaa
def process_name(methodname):
    name_str = ''
    last_name = str(methodname)
    for i in range(len(last_name)):
        if (last_name[i] <= 'z' and last_name[i] >= 'a') or (last_name[i] <= '9' and last_name[i] >= '0'):
            name_str += last_name[i]
        elif last_name[i] <= 'Z' and last_name[i] >= 'A':
            name_str += ' ' + last_name[i].lower()
    return name_str

def train_embedding(wordlist):
    client = BertClient(ip='127.0.0.1')
    sample_token = []
    for item in wordlist:
        sample_token.append(word_tokenize(item))
    sample_vec = client.encode(sample_token, is_tokenized=True)
    return sample_vec

def test_embedding(test):
    client = BertClient(ip='127.0.0.1')
    test_token = word_tokenize(test)
    test_vec = client.encode([test_token], is_tokenized=True)
    return test_vec

def get_similarity(embedding, testvec):
    if len(embedding) == 0:
        return []
    sim = cosine_similarity(embedding, testvec.reshape(1,-1))
    final_sim = []
    for item in sim:
        final_sim.append(item[0])
    return final_sim

# from the calculated similarity to choose the top similar definition
def choose_topk_input(similarity, k):
    print("here top k: " + str(k))
    tmp_similarity = []
    for i in range(len(similarity)):
        tmp_similarity.append((i, similarity[i]))

    choice = sorted(tmp_similarity, key=lambda x:x[1], reverse=True)
    if k > len(choice):
        k = len(choice)
    return choice[0:k]

# output to the seed file based on different type
# just like the format as the initial seed file
def output(topk, func_list, input_list, target_input, level):
    if len(topk) == 0:
        return {"func_name": "noinputs!!", "func_inputs": []}
    target_input_json = json.loads(target_input)
    target_func = target_input_json["name"]
    input_pool = []
    # target_input means the function which we want to get the input
    try:
        for idx,_ in topk:
            inputs = input_list[idx]
            for item in inputs:
                input_pool.append(item)
            
    except:
        print("error")
        return {"func_name": target_func, "func_inputs": input_pool}
    # determine a path to output
    return {"func_name": target_func, "func_inputs": input_pool}



        

def main():

    fd = open('../data/exp1/all_funcs_need.json') # need to modify the route
    content = fd.read()
    original_func = json.loads(content)[59876:]
    
    word_list,input_list = read_funcdef_from_file('../data/exp1/sample.json') # need to modify the route
    # word_list, input_list = read_funcdef_from_file('D:\\EtherscanInfo\\final_without_TSE.json')
    
    filepath = '../data/exp1/test.json'# target abi
    outputpath = '../data/exp1/' # output file path
    # target_functions is full of functions in the target abi
    # {"name" : name, "inputs" : [{"name" : inputname, "type" : type}]}
    target_functions = read_targets_from_abi(filepath)
    # embedding = train_embedding(process_functions(word_list))
    # fd = open(outputpath + 'embedding.txt', 'w+')
    # fd.write(json.dumps({'res':embedding.tolist()}))
    
    fd = open('../data/exp1/embedding.txt', 'r') # need to modify the route
    content = fd.read()
    embedding_matrix = json.loads(content)
    res = []
    # item = target_functions[0]
    # print(len(embedding_matrix["res"]))
    cnt = 4000 # 0 2000 4000
    same_in_level1 = 0
    same_in_level2 = 0
    same_in_level3 = 0
    for item in target_functions[4000:]: # [0:2000] [2000:4000] [4000:]
        if cnt == 4282: # bad
            cnt += 1
            continue
        print(cnt)
        print(json.loads(item)["name"])
        test_word, input_type_list = process_input(item)
        first_level_funcs,first_level_inputs,first_level_embedding,second_level_funcs,second_level_inputs,second_level_embedding,third_level_funcs,\
        third_level_inputs,third_level_embedding = trim_func(word_list, input_type_list, input_list, embedding_matrix["res"])
        test_vec = test_embedding(test_word)
        first_similarity = get_similarity(first_level_embedding, test_vec)
        first_level_top = choose_topk_input(first_similarity, 10)
        output1 = output(first_level_top, first_level_funcs, first_level_inputs, item, "first_level")
        second_similarity = get_similarity(second_level_embedding, test_vec)
        second_level_top = choose_topk_input(second_similarity, 10)
        output2 = output(second_level_top, second_level_funcs, second_level_inputs, item, "second_level")
        third_similarity = get_similarity(third_level_embedding, test_vec)
        third_level_top = choose_topk_input(third_similarity, 10)
        output3 = output(third_level_top, third_level_funcs, third_level_inputs, item, "third_level")
        
        original_input = original_func[cnt]["input"]
        for item1 in output1['func_inputs']:
            length = 0
            just_addr = 0
            for i in range(len(original_input)):
                types = original_func[cnt]['function']['inputs'][i]['type']
                if 'address' in types:
                    length += 1
                    just_addr += 1
                    continue
                if original_input[i] in item1:
                    length += 1
            if  length == len(original_input) and length > 0 and length != just_addr:
                print(original_input)
                print(original_func[cnt])
                print(item1)
                print('got it in first level!')
                print('\n')
                same_in_level1 += 1
                break

        for item1 in output2['func_inputs']:
            length = 0
            just_addr = 0
            for i in range(len(original_input)):
                types = original_func[cnt]['function']['inputs'][i]['type']
                if 'address' in types:
                    length += 1
                    just_addr += 1
                    continue
                if original_input[i] in item1:
                    length += 1
            if  length == len(original_input) and length > 0 and length != just_addr:
                print(original_input)
                print(original_func[cnt])
                print(item1)
                print('got it in second level!')
                print('\n')
                same_in_level2 += 1
                break

        for item1 in output3['func_inputs']:
            length = 0
            just_addr = 0
            for i in range(len(original_input)):
                types = original_func[cnt]['function']['inputs'][i]['type']
                if 'address' in types:
                    length += 1
                    just_addr += 1
                    continue
                if original_input[i] in item1:
                    length += 1
            if  length == len(original_input) and length > 0 and length != just_addr:
                print(original_input)
                print(original_func[cnt])
                print(item1)
                print('got it in third level!')
                print('\n')
                same_in_level3 += 1
                break

        # res.append({'1':output1, '2':output2, '3':output3})
        cnt += 1
    print("level1:" + str(same_in_level1))
    # res.append(output2)
    print("level2:" + str(same_in_level2))
    # res.append(output3)
    print("level3:" + str(same_in_level3))
    # print(res)
    
    # with open(outputpath + "/threelevel_1_1.txt", "w+") as fp:
    #     fp.write(json.dumps({"res": res}))
    #     fp.close()
    # with open(outputpath + "/threelevel_2_1.txt", "w+") as fp:
    #     fp.write(json.dumps({"res": res[2000:4000]}))
    #     fp.close()
    # with open(outputpath + "/threelevel_3_1.txt", "w+") as fp:
    #     fp.write(json.dumps({"res": res[4000:]}))
    #     fp.close()

if __name__ == '__main__':
    main()
    exit(0)
