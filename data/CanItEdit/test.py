import json
import subprocess
import tempfile
import os
import re
from util import (
    read_json,
    save_json
)
from tqdm import tqdm

from icecream import install
install()

def test_completions(data_dict: dict) -> bool:
    try:
        completion_code = data_dict["completions"][0]
        test_code = data_dict["tests"]
        ic(completion_code, test_code)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(completion_code)
            temp_file.write("\n\n")
            temp_file.write(test_code)
            temp_file_path = temp_file.name
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=False,
            text=False,
            check=False,
            timeout=10
        )
        if result.returncode==0:
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        return False
    except Exception as e:
        return False
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


test_path = "result/example.jsonl"
test_json = read_json(test_path,True)
pass_result = 0
throughput = 0
for data in tqdm(test_json):
    throughput += data['throughput']
    if 'pass' in data:
        if data['pass']:
            pass_result+=1
        continue
    idx = data["completions"][0].find('``')
    data["completions"][0] =  data["completions"][0][:idx-1] if idx != -1 else data["completions"][0]

    if test_completions(data)==True:
        pass_result+=1
        data['pass'] = True
    else:
        data['pass'] = False
print(f"pass@1: {pass_result/len(test_json)}")
print(f"throughput: {throughput/len(test_json)}")
###save_result###
