# DeepLearning For Regression With DNN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-13.2-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

<p align="center">
  基于 PyTorch 的深度学习回归实战系列<br/>
  以加州住房价格数据集为例，从基础 MLP 到自定义损失函数、自定义网络层，循序渐进掌握 DNN 回归任务核心技巧。
</p>

---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [Notebook 详解](#notebook-详解)
- [快速开始](#快速开始)
- [学习路线](#学习路线)

---

## 项目简介

本项目是一个 **循序渐进的 PyTorch 回归实战教学系列**，以 scikit-learn 内置的 **加州住房价格（California Housing）** 数据集为统一实验载体，涵盖以下三个递进主题：

- **浅层神经网络（Shallow MLP）**：建立完整的 PyTorch 训练流程认知
- **自定义损失函数**：继承 `nn.Module` 手动实现 MSE，理解损失函数的设计范式
- **自定义网络层**：使用 `nn.Parameter` 从零实现全连接层，深入理解层的内部机制

每个 Notebook 均可独立运行，包含完整的 **数据处理 → 模型构建 → 训练 → 评估** 闭环。

---

## 项目结构

```
DeepLearningForRegressionWithDNN/
│
├── 1.regression_s(shallow)nn.ipynb      # 浅层神经网络回归基础
├── 2.regression_CustomizedLoss.ipynb    # 自定义损失函数（MSE）
├── 3.regression_CustomizedLayer.ipynb   # 自定义全连接层（CustomizedLinear）
│
├── requirements.txt                     # 项目直接依赖清单
├── LICENSE
└── README.md
```

---

## Notebook 详解

### 📘 1. 浅层神经网络回归 `1.regression_s(shallow)nn.ipynb`

> 建立从数据到训练评估的完整 PyTorch 回归流程。

| # | 章节 | 核心内容 |
|:-:|------|---------|
| 一 | 环境配置 | 库导入、GPU / CPU 设备检测与配置 |
| 二 | 数据准备 | California Housing 加载与探索、训练 / 验证 / 测试集划分、`StandardScaler` 标准化、自定义 `Dataset`、`DataLoader` 构建 |
| 三 | 模型构建 | `NeuralNetwork` 网络结构定义、参数量统计、`EarlyStopCallback` 早停回调、评估与训练函数封装 |
| 四 | 模型评估 | 测试集 MSE 评估、学习曲线可视化 |

---

### 📗 2. 自定义损失函数 `2.regression_CustomizedLoss.ipynb`

> 继承 `nn.Module` 手动实现 MSE 损失，并与内置实现进行数值一致性验证。

| # | 章节 | 核心内容 |
|:-:|------|---------|
| 1 | 环境初始化 | 库导入与设备配置 |
| 2 | 准备数据 | 数据集加载、划分、标准化、`Dataset` / `DataLoader` |
| 3 | 构建回归模型 | 标准全连接网络定义 |
| 4 | 训练辅助组件 | `EarlyStopCallback` 早停、验证评估函数、**自定义 MSE 损失函数**、与 PyTorch 内置 MSE 一致性对比验证 |
| 5 | 模型训练 | 完整训练循环 |
| 6 | 可视化 | 训练 / 验证损失曲线绘制 |
| 7 | 测试集评估 | 最终模型在测试集上的 MSE |

**核心知识点**

```python
class CustomizedMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)
```

---

### 📙 3. 自定义全连接层 `3.regression_CustomizedLayer.ipynb`

> 使用 `nn.Parameter` 从零实现 `CustomizedLinear`，替代内置 `nn.Linear` 并组合成完整回归网络。

| # | 章节 | 核心内容 |
|:-:|------|---------|
| 1 | 环境初始化 | 库导入与设备配置 |
| 2 | 准备数据 | 数据集加载、划分、标准化、`Dataset` / `DataLoader` |
| 3 | 构建回归模型 | **`CustomizedLinear` 自定义全连接层**、基于 `CustomizedLinear` 搭建网络、模型参数查看 |
| 4 | 训练辅助组件 | `EarlyStopCallback` 早停、验证评估函数 |
| 5 | 模型训练 | 完整训练循环 |
| 6 | 可视化 | 训练 / 验证损失曲线绘制 |
| 7 | 测试集评估 | 最终模型在测试集上的 MSE |

**核心知识点**

```python
class CustomizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 使用 nn.Parameter 注册可训练参数，参与自动微分
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 实现线性变换：y = x @ W^T + b
        return x @ self.weight.T + self.bias
```


---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/<your-username>/DeepLearningForRegressionWithDNN.git
cd DeepLearningForRegressionWithDNN

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Jupyter
jupyter notebook
```

然后按编号顺序依次打开并运行三个 Notebook 即可。

---

## 学习路线

```
1. regression_s(shallow)nn
   ├─ 建立完整 PyTorch 训练流程认知
   └─ 掌握 Dataset / DataLoader / EarlyStopping 模式
            │
            ▼
2. regression_CustomizedLoss
   ├─ 理解损失函数的 nn.Module 封装范式
   └─ 学会验证自定义实现与内置实现的等价性
            │
            ▼
3. regression_CustomizedLayer
   ├─ 掌握 nn.Parameter 注册可训练参数
   └─ 从线性代数角度理解全连接层的本质
```

---

## License

本项目基于 [MIT License](./LICENSE) 开源。
