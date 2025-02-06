# oat-zero

One of the most inspiring results from DeepSeek-R1-Zero is the occurrence of “Aha moment” through pure reinforcement learning (RL). At the Aha moment, the model learns emergent skills such as self-reflection, which helps it to conduct in-context search to solve complex reasoning problems.

Within only a few days after R1-Zero's release, several projects independently “reproduced” R1-Zero-like training on smaller scales (e.g., 1B to 7B) and all observed the Aha moment, which is typically measured by ***a sudden increase of model response length***. We follow their settings to scrutinize the R1-Zero-like training process, and share the following findings:



## Table of Contents

- [Quick Start](#quick-start)
- [License](#license)

## Introduction

[placeholder: some introduction]

## Quick Start

### Installation

Before installing `oat-zero`, please install `oat` following the instructions in https://github.com/sail-sg/oat. 

Once `oat` is installed, install `oat-zero` by running the following command:

```console
git clone https://github.com/sail-sg/oat-zero.git
cd oat-zero
pip install -e .
```

### Evaluating Self-Reflection Behavior

To evaluate the self-reflection behavior of the base model, run:

```console
# example of analyzing Qwen/Qwen2.5-Math-7B
bash analysis/script.sh Qwen/Qwen2.5-Math-7B
```

Simply run the script with the model name as the argument.


### Reproducing Oat-Zero

TODO: add links to sec 3.1 and 3.2

To reproduce the results of Oat-Zero at Sec. 3.1, run: 

```console
bash training/run_grpo_7b.sh
```

To reproduce the results at Sec. 3.2, please follow the instructions in [simpleRL/train](https://github.com/hkust-nlp/simpleRL-reason/tree/main/train). 

## Acknowledgement


## Citation

TODO: add url

```bibtex
@misc{liu2025oatzero,
  title={There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study},
  author={Zichen Liu and Changyu Chen and Wenjun Li and Tianyu Pang and Chao Du and Min Lin},
  year={2025},
  howpublished={\url{https://notion.site}},
  note={Notion Blog},
}
```


## License

`oat-zero` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
