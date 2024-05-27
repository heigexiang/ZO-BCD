import json, re

with open('/mnt/data/xue.w/yutong/data/grade_school_math/data/test.jsonl', 'r') as f:
    data_test = f.readlines()
    data_test = [json.loads(d) for d in data_test]

with open('/mnt/data/xue.w/yutong/data/grade_school_math/data/train.jsonl', 'r') as f:
    data_train = f.readlines()
    data_train = [json.loads(d) for d in data_train]

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return "[invalid]"

res = 0

for data in data_train + data_test:
    answer = extract_answer(data['answer'])
    try:
        floatanswer = float(answer)
        res += floatanswer
        # check if the answer is a valid number
    except:
        print("Invalid answer: ", answer)
        print(data)
        print()
        continue
print(res, res is float('nan'))