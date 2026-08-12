<h1 align="center">BEHAVIOR-1K</h1>

![BEHAVIOR-1K](./docs/assets/readme_splash_logo.png)

**BEHAVIOR-1K** is a comprehensive simulation benchmark for testing embodied AI agents on 1,000 everyday household activities. This monolithic repository provides everything needed to train and evaluate agents on human-centered tasks like cleaning, cooking, and organizing — activities selected from real human time-use surveys and preference studies.

***Check out our [main website](https://behavior.stanford.edu/) for more details!***

# 🛠️ Installation

BEHAVIOR-1K provides an installation script that handles all dependencies and components. The script supports modular installation, allowing you to install only the components you need.

Please check out [Installation Guide](https://behavior.stanford.edu/getting_started/installation.html) for more details!

## 2026 PI0.5 加速评测

`speedup_eval` 分支上的双环境、batch-2、持久 server 和多 GPU 均衡调度方案，以及冠军模型专用 correction rules 的接入边界，见 [PI0.5 2026 加速评测说明](docs/pi05_behavior_2026_eval_acceleration.md)。

推荐入口：

```bash
bash run_pi05_behavior_2026_eval_chunk_balance.sh
```

接入其他模型前请先阅读文档中的“冠军方案的环境侧适配”；冠军 PI0.5 correction rules 默认在 env/evaluator 侧启用，不应直接应用到其他模型。

## 📄 Citation

```bibtex
@article{li2024behavior1k,
    title   = {BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
    author  = {Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Martín-Martín and Chen Wang and Gabrael Levine and Wensi Ai and Benjamin Martinez and Hang Yin and Michael Lingelbach and Minjune Hwang and Ayano Hiranaka and Sujay Garlanka and Arman Aydin and Sharon Lee and Jiankai Sun and Mona Anvari and Manasi Sharma and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Yunzhu Li and Silvio Savarese and Hyowon Gweon and C. Karen Liu and Jiajun Wu and Li Fei-Fei},
    journal = {arXiv preprint arXiv:2403.09227},
    year    = {2024}
}
```
