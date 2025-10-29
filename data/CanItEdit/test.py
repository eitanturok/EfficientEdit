import subprocess
import tempfile
import os
from argparse import ArgumentParser

from tqdm import tqdm

from util import read_json

from icecream import install
install()

def test_completions(data_dict: dict) -> bool:
    try:
        completion_code = data_dict["completions"]
        test_code = data_dict["tests"]
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

def main(args):
    test_json = read_json(args.output_path,True)
    pass_result = 0
    # throughput = 0
    for data in tqdm(test_json):
        # throughput += data['throughput']
        if 'pass' in data:
            if data['pass']:
                pass_result+=1
            continue
        idx = data["completions"].find('``')
        data["completions"] =  data["completions"][:idx-1] if idx != -1 else data["completions"]

        if test_completions(data)==True:
            pass_result+=1
            data['pass'] = True
        else:
            data['pass'] = False
    print(f"pass@1: {pass_result/len(test_json)}")
    # print(f"throughput: {throughput/len(test_json)}")
    ###save_result###

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-path", type=str, default="result/example.jsonl")
    args = parser.parse_args()
    main(args)
