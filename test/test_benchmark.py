import json
import subprocess

from atgen.utils.get_last_workdir import get_last_workdir


def check_iteration(workdir, n_iter: int):
    with open(workdir / ("iter_" + str(n_iter)) / "metrics.json") as f:
        result = json.load(f)
    assert result["rouge1"] >= 0.17
    assert result["rougeL"] >= 0.13


def exec_bash(s):
    return subprocess.run(s, shell=True)


def test_just_works():
    exec_result = exec_bash(
        "HYDRA_CONFIG_NAME=test run-al +debug=false +save_model=false"
    )
    assert exec_result.returncode == 0


def test_normal_results():
    workdir = get_last_workdir()
    check_iteration(workdir, 0)
    check_iteration(workdir, 1)


# def test_openai_labelling():
#     exec_result = exec_bash(
#         "HYDRA_CONFIG_NAME=test python3 run_scripts/run_active_learning.py +debug=false labeller=api_llm labeller.api_key_path=outputs/api_key_openai.key al.num_iterations=1 al.query_size=10"
#     )
#     assert exec_result.returncode == 0

#     workdir = get_last_workdir()
#     check_zero_iteration(workdir)

#     with open(workdir / "1" / "results.json") as f:
#         result = json.load(f)
#     assert result["bleu"] >= 0.105
#     assert result["rouge1"] >= 0.31
#     assert result["rouge2"] >= 0.09
#     assert result["rougeL"] >= 0.26

# def test_anthropic_labelling():
#     exec_result = exec_bash(
#         "HYDRA_CONFIG_NAME=test python3 run_scripts/run_active_learning.py +debug=false labeller=api_llm labeller.api_key_path=outputs/api_key.key labeller.provider=anthropic labeller.parameters.model=claude-3-5-sonnet-20240620 al.num_iterations=1 al.query_size=10"
#     )
#     assert exec_result.returncode == 0

#     workdir = get_last_workdir()
#     check_zero_iteration(workdir)

#     with open(workdir / "1" / "results.json") as f:
#         result = json.load(f)
#     assert result["bleu"] >= 0.11
#     assert result["rouge1"] >= 0.31
#     assert result["rouge2"] >= 0.09
#     assert result["rougeL"] >= 0.26
