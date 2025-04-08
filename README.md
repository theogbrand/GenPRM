<div align="center">

# GenPRM

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.00891) [![Website](https://img.shields.io/badge/Project_Page-000acc?style=for-the-badge&logo=githubpages&logoColor=000&logoColor=white)](https://ryanliu112.github.io/GenPRM)  [![Github](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/RyanLiu112/GenPRM)  [![HuggingFace](https://img.shields.io/badge/HugggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/GenPRM/genprm-67ee4936234ba5dd16bb9943)  [![Awesome Process Reward Models](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/RyanLiu112/Awesome-Process-Reward-Models)

</div>

<div align="center">
  <p>
    <a href="#-news" style="text-decoration: none; font-weight: bold;">🔔 News</a> •
    <a href="#-method" style="text-decoration: none; font-weight: bold;">👀 Method</a> •
    <a href="#-results" style="text-decoration: none; font-weight: bold;">🏆 Results</a>
  </p>
  <p>
    <a href="#-getting-started" style="text-decoration: none; font-weight: bold;">🚀 Getting Started</a> •
    <a href="#-citation" style="text-decoration: none; font-weight: bold;">📝 Citation</a> •
    <a href="#-acknowledgement" style="text-decoration: none; font-weight: bold;">💡 Acknowledgement</a>
  </p>
</div>

## 🎯 Overview

<img src="./static/images/fig_head.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="fig_head">

We propose **GenPRM**, a strong generative process reward model with the following features:

- performing explicit **CoT reasoning** and **code verfication** before providing the process judgment;
- improving Monte Carlo estimation and hard label with **Relative Progress Estimation (RPE)**;
- supporting GenPRM **test-time scaling** in a parallel manner with majority voting;
- supporting policy model test-time scaling with GenPRM as **verifiers** or **critics**.

We will release all code, model, and data, including:

- GenPRM with parameters of 1.5B, 7B, 14B, 32B, and 70B (ongoing);
- 23K training data from MATH dataset;
- all details including solution generation, Monte Carlo estimation, RPE, model training and inference (ongoing).

<img src="./static/images/comparison.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="comparison">

## 🔔 News

- **[2025-04-06]** ✨ The evaluation code and [GenPRM-32B](https://huggingface.co/GenPRM/GenPRM-32B) are available.
- **[2025-04-05]** ✨ The inference code is available.
- **[2025-04-03]** ✨ Our models ([GenPRM-1.5B](https://huggingface.co/GenPRM/GenPRM-1.5B) & [GenPRM-7B](https://huggingface.co/GenPRM/GenPRM-7B)) and training data are released on [HuggingFace](https://huggingface.co/collections/GenPRM/genprm-67ee4936234ba5dd16bb9943).
- **[2025-04-01]** 📄 Our paper is released on [arXiv](https://arxiv.org/abs/2504.00891).

## 👀 Method

Our framework:

<img src="./static/images/framework2.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="framework2">

## 🏆 Results

### ProcessBench

<img src="./static/images/main_processbench.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="main_processbench">

### Best-of-N

<img src="./static/images/main_bon.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="main_bon">

### Critique Refinement

<img src="./static/images/critic.png" alt="" style="width: 100%; max-width: 1000px; margin-top: 20px; margin-bottom: 10px;" id="fig_head">

## 🚀 Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/RyanLiu112/GenPRM.git
cd GenPRM/src
```

Create a new conda environment and install the dependencies:

```bash
conda create -n GenPRM python=3.10
conda activate GenPRM
pip install -r requirements.txt
```

### Examples & Demos

Try GenPRM in action with:

- **Interactive Jupyter Notebook**: [demo.ipynb](src/example/demo.ipynb) (quick start of GenPRM inference)
- **Process Supervision Cases**: [Case 1](src/example/case1.md) | [Case 2](src/example/case2.md)

For a quick start, you can use [gemprm_inference](src/prm_evaluation/gemprm_inference.py) module to implement model inference:

```python
from prm_evaluation.genprm_inference import GenPRM, CodeExecutor

genprm = GenPRM('GenPRM/GenPRM-7B')

messages = [ 
    {"role": "system", "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."}, 
    {"role": "user", "content": "Question: Jo adds up all the positive integers from 1 to 100. Kate does a similar thing with the first 100 positive integers; however, she first rounds every integer to its nearest multiple of 10 (rounding 5s up) and then adds the 100 values. What is the positive difference between Jo's sum and Kate's sum?\n\nFirst, we need to calculate Jo's sum, which is the sum of all positive integers from 1 to 100. This can be directly computed using the formula for the sum of the first \\(n\\) positive integers, which is \\(\\frac{n(n+1)}{2}\\). For \\(n = 100\\), Jo's sum is \\(\\frac{100 \\cdot 101}{2} = 5050\\)."}, 
]
code_executor = CodeExecutor()
output, reward = genprm.inference(messages, cur_step=1, code_executor=code_executor)
print("Model output for the first solution step: " + output[0])
print(reward)
```

### ProcessBench / Best-of-N evaluation

Split the dataset into individual shards (require `steps` and `problem` fields)

```bash
# example of processbench
python utils/split_dataset.py \
    --dataset "Qwen/ProcessBench" \
    --split_dir "_data/split_input/ProcessBench"
```

Generate step-by-step outputs of PRM

```bash
# example of processbench
python prm_evaluation/prm_evaluate.py \
    --reward_name_or_path "GenPRM/GenPRM-7B" \
    --data_path "_data/split_input/ProcessBench" \
    --split_out "_output/split_output/ProcessBench" \
    --analyze \
    --verify \
    --execute
```

### Critique-refinement

Execute policy refinement based on GenPRM's split output

```bash
python prm_evaluation/policy_refine.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "_output/split_output/..."\
    --split_out "_output/split_refine/..."
```



> [!NOTE]
> Our mathematical expression evaluation code is based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math). For a more powerful evaluator, please refer to this repository: [Math-Verify](https://github.com/huggingface/Math-Verify).



## 📝 Citation

If you find this work helpful, please kindly cite our paper:

```bibtex
@article{zhao2025genprm,
    title   = {GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning},
    author  = {Jian Zhao and Runze Liu and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    journal = {arXiv preprint arXiv:2504.00891},
    year    = {2025}
}
```

Our collection of PRMs in [Awesome-Process-Reward-Models](https://github.com/RyanLiu112/Awesome-Process-Reward-Models):

```bibtex
@misc{Awesome-Process-Reward-Models,
    title        = {Awesome Process Reward Models},
    author       = {Runze Liu and Jian Zhao and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    howpublished = {\url{https://github.com/RyanLiu112/Awesome-Process-Reward-Models}},
    note         = {GitHub repository},
    year         = {2025}
}
```

Our recent work on LLM test-time scaling with PRMs:

```bibtex
@article{liu2025can,
    title   = {Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling},
    author  = {Runze Liu and Junqi Gao and Jian Zhao and Kaiyan Zhang and Xiu Li and Biqing Qi and Wanli Ouyang and Bowen Zhou},
    journal = {arXiv preprint arXiv:2502.06703},
    year    = {2025}
}
```

## 💡 Acknowledgement

The model training is based on [axolotl](https://github.com/axolotl-ai-cloud/axolotl) and [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm). The mathematical evaluation code is based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).


