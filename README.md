<div align="center">

# GenPRM



</div>

<div align="center">
  <p>
    <a href="#-news" style="text-decoration: none; font-weight: bold;">🔔 News</a> •
    <a href="#-tts-methods" style="text-decoration: none; font-weight: bold;">👀 Method</a> •
    <a href="#-results" style="text-decoration: none; font-weight: bold;">🏆 Results</a>
  </p>
  <p>
    <a href="#-getting-started" style="text-decoration: none; font-weight: bold;">🚀 Getting Started</a> •
    <a href="#-citation" style="text-decoration: none; font-weight: bold;">📝 Citation</a> •
    <a href="#-acknowledgement" style="text-decoration: none; font-weight: bold;">💡 Acknowledgement</a>
  </p>
</div>



## 🔔 News

<!-- - **[2025-03]** 📄 Our paper is released on [arXiv](https://arxiv.org/abs/2503.xxxxx). -->

TBD

## 👀 Method

TBD

## 🏆 Results

TBD

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

TBD

> [!NOTE]
> Our mathematical expression evaluation code is based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math). For a more powerful evaluator, please refer to this repository: [Math-Verify](https://github.com/huggingface/Math-Verify).




## 📝 Citation

If you find this work helpful, please kindly cite our paper:

```bibtex
@article{zhao2025genprm,
    title   = {GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning},
    author  = {Jian Zhao and Runze Liu and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    journal = {arXiv preprint arXiv:2503.xxxxx},
    year    = {2025}
}
```

You can also check out our collection of PRMs in [Awesome-Process-Reward-Models](https://github.com/RyanLiu112/Awesome-Process-Reward-Models):

```bibtex
@misc{Awesome-Process-Reward-Models,
    title   = {Awesome Process Reward Models},
    author  = {Runze Liu and Jian Zhao and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    url     = {https://github.com/RyanLiu112/Awesome-Process-Reward-Models},
    note    = {GitHub Repository},
    year    = {2025}
}
```

Our recent work on LLM test-time scaling with PRMs on mathematical tasks:

```bibtex
@article{liu2025can,
    title   = {Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling},
    author  = {Runze Liu and Junqi Gao and Jian Zhao and Kaiyan Zhang and Xiu Li and Biqing Qi and Wanli Ouyang and Bowen Zhou},
    journal = {arXiv preprint arXiv:2502.06703},
    year    = {2025}
}
```

## 💡 Acknowledgement

GenPRM training is based on [axolotl](https://github.com/axolotl-ai-cloud/axolotl) and [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm). The evaluation code is based on [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).


