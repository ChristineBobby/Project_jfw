# 题目 2 项目核心指导文档：基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

版本日期：2026-06-04  
项目路径：`/home/xiaobo.xia/JiafengWu/code_folder/area1/Project_jfw`  
选题：题目 2，基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

## 目录

- [0. 本文档定位](#sec-0)
- [1. 已核查事实与来源边界](#sec-1)
  - [1.1 课程 PDF 硬性要求](#sec-1-1)
  - [1.2 数据集事实](#sec-1-2)
  - [1.3 已核查论文与可采用结论](#sec-1-3)
  - [1.4 严格限制](#sec-1-4)
- [2. 总体课设架构](#sec-2)
  - [2.1 推荐主线](#sec-2-1)
  - [2.2 主实验选择单位](#sec-2-2)
  - [2.3 推荐数据划分](#sec-2-3)
- [3. 数据字段与任务定义](#sec-3)
  - [3.1 输入字段](#sec-3-1)
  - [3.2 输出指标](#sec-3-2)
- [4. 特征与模型选择](#sec-4)
  - [4.1 推荐特征提取器](#sec-4-1)
  - [4.2 MLP 主模型](#sec-4-2)
- [5. PC-RAS-Coreset 方法设计](#sec-5)
  - [5.1 冗余定义](#sec-5-1)
  - [5.2 脑启发映射](#sec-5-2)
  - [5.3 帧级分数](#sec-5-3)
  - [5.4 episode 级选择](#sec-5-4)
  - [5.5 帧级 10% 扩展](#sec-5-5)
- [6. 实验矩阵](#sec-6)
  - [6.1 必做实验](#sec-6-1)
  - [6.2 方法消融](#sec-6-2)
  - [6.3 额外研究实验](#sec-6-3)
- [7. 推荐代码目录结构](#sec-7)
- [8. 逐步执行流程](#sec-8)
  - [Step 0：环境准备](#sec-8-0)
  - [Step 1：数据集检查](#sec-8-1)
  - [Step 2：固定划分](#sec-8-2)
  - [Step 3：离线特征提取](#sec-8-3)
  - [Step 4：随机 baseline 选择](#sec-8-4)
  - [Step 5：计算 PC-RAS 分数](#sec-8-5)
  - [Step 6：episode 级 PC-RAS-Coreset](#sec-8-6)
  - [Step 7：训练 PC-RAS MLP](#sec-8-7)
  - [Step 8：消融实验](#sec-8-8)
  - [Step 9：可视化](#sec-8-9)
  - [Step 10：报告撰写](#sec-8-10)
- [9. 报告中的核心论点](#sec-9)
  - [9.1 推荐论点](#sec-9-1)
  - [9.2 独特价值](#sec-9-2)
- [10. 风险与应对](#sec-10)
- [11. 推荐配置草案](#sec-11)
- [12. 最小可交付版本与增强版本](#sec-12)
  - [12.1 最小可交付版本](#sec-12-1)
  - [12.2 高质量增强版本](#sec-12-2)
- [13. 预期结果格式](#sec-13)
- [14. 工作优先级](#sec-14)
- [15. 参考链接清单](#sec-15)

---

<a id="sec-0"></a>

## 0. 本文档定位

本文档是本项目的核心实施依据，目标不是简单跑通一个随机 10% baseline，而是在课程题目要求之上，形成一个有解释性、有消融、有可视化、有认知机制映射的轻量级 VLA 数据修剪课设。

项目最终要回答的主问题：

> 在 ALOHA Sim Transfer Cube 数据集上，能否通过脑启发的“预测编码式时序惊奇度 + RAS 式任务效用 + 分布覆盖去冗余”选择 10% 高价值数据，使轻量 MLP 的 7 自由度动作预测 MSE 优于随机 10% 轨迹 baseline，并能解释为什么这些样本更有价值？

本项目推荐方法命名为：

> **PC-RAS-Coreset**：Predictive-Coding and RAS inspired Coreset Selection.

---

<a id="sec-1"></a>

## 1. 已核查事实与来源边界

<a id="sec-1-1"></a>

### 1.1 课程 PDF 硬性要求

本地 PDF：

`/home/xiaobo.xia/JiafengWu/code_folder/area1/Project_jfw/2026春季认知工程考查试卷.pdf`

PDF 共 5 页，已用 `pdfinfo` 和 `pdftotext -layout` 完整提取。题目 2 要求如下：

- 数据集：ALOHA Sim Transfer Cube (Human Demonstrations)。
- 数据来源：Hugging Face LeRobot 官方库。
- baseline：随机抽取 10% 轨迹样本。
- 特征：使用冻结预训练轻量视觉模型，例如 ResNet-18 或 CLIP，离线提取图像特征。
- 任务形式：构建 `[视觉特征 + 语言指令] -> [7 自由度机械臂动作]` 的特征回归数据集。
- 模型：训练轻量级 MLP。
- 指标：报告动作预测 MSE。
- 核心任务：设计自动化数据修剪算法，从全量数据集中筛选 10% coreset，并与随机 10% baseline 对比。
- 报告重点：背景机制调研、冗余定义、核心集选择算法设计、对比实验结果分析、源代码和清晰注释。
- 提交截止：PDF 写明 2026 年 6 月 20 日前收齐研究报告。

<a id="sec-1-2"></a>

### 1.2 数据集事实

官方数据集页：

- Hugging Face dataset: <https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human>
- 官方 `meta/info.json` 搜索结果和 Hugging Face 页面显示：
  - `total_episodes = 50`
  - `total_frames = 20000`
  - `fps = 50`
  - `total_tasks = 1`
  - `observation.images.top` 为视频特征，shape 为 `[480, 640, 3]`
  - `observation.state` 为 14 维
  - `action` 为 14 维
  - 动作和状态电机名称分为 left/right 两组，每组 7 维
  - 数据集页面 viewer 显示 `episode_index` 范围 0 到 49，`frame_index` 范围 0 到 399，因此每个 episode 约 400 帧
  - 页面标签包含 Robotics、Tabular、Time-series、Video、parquet、MIT license

LeRobot 数据格式文档：

- <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- LeRobotDataset v3.0 是机器人学习数据的标准化格式，提供多模态时序数据、sensorimotor signals、多相机视频和 metadata。
- 文档给出 `LeRobotDataset(repo_id)` 的加载方式，并说明样本以 PyTorch tensor 字典形式返回，可接入 `torch.utils.data.DataLoader`。

<a id="sec-1-3"></a>

### 1.3 已核查论文与可采用结论

只把能在 arXiv、Hugging Face、PyTorch 官方文档中定位的信息纳入本设计。

| 方向 | 来源 | 可采用结论 |
| --- | --- | --- |
| ACT / ALOHA 基础 | Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, arXiv:2304.13705, <https://arxiv.org/abs/2304.13705> | ACT 用 action chunking 学习动作序列，是 ALOHA / 双臂精细操作的重要基础。课程数据集也关联此 arXiv。 |
| 数据修剪理论 | Sorscher et al., *Beyond neural scaling laws*, NeurIPS 2022, arXiv:2206.14486, <https://arxiv.org/abs/2206.14486> | 高质量 data pruning metric 可以在减少数据规模时保持甚至改善 scaling 行为，给本题“10% 高质量数据”提供理论动机。 |
| 经典 coreset | Sener & Savarese, *Active Learning for CNNs: A Core-Set Approach*, arXiv:1708.00489, <https://arxiv.org/abs/1708.00489> | 可将 subset selection 定义为选择一组点，使在该子集上学习的模型能覆盖剩余数据；几何覆盖思想适合作为分布冗余过滤。 |
| 预测编码 | Millidge, Seth & Buckley, *Predictive Coding: a Theoretical and Experimental Review*, arXiv:2107.12979, <https://arxiv.org/abs/2107.12979> | 预测编码把大脑功能解释为最小化对世界生成模型的预测误差，本项目据此把“预测误差/状态变化”定义为高价值时序信号。 |
| CLIP | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, arXiv:2103.00020, <https://arxiv.org/abs/2103.00020> | CLIP 通过图文对学习通用视觉表征，适合作为冻结视觉-语言特征提取器。 |
| ResNet-18 | PyTorch torchvision resnet18 文档, <https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html> | torchvision 提供 ResNet-18 和 ImageNet-1K 预训练权重，输出可作为轻量视觉特征。 |
| OpenVLA | Kim et al., *OpenVLA*, arXiv:2406.09246, <https://arxiv.org/abs/2406.09246> | OpenVLA 是 7B 参数开源 VLA，使用 DINOv2 + SigLIP 视觉特征和 970k 机器人演示；论文强调 LoRA/量化等消费级 GPU 友好的效率路径。 |
| 2025 轻量 VLA | Shukor et al., *SmolVLA*, arXiv:2506.01844, <https://arxiv.org/abs/2506.01844> | SmolVLA 强调小模型、单 GPU 训练、消费级 GPU/CPU 部署，说明课设做“轻量/高效”是当前 VLA 趋势。 |
| 2025/2026 通用 VLA | *pi0*, arXiv:2410.24164, <https://arxiv.org/abs/2410.24164>; *pi0.5*, arXiv:2504.16054, <https://arxiv.org/abs/2504.16054> | pi0 使用 VLM + flow matching 构建通用机器人控制模型；pi0.5 强调用多机器人、多任务、语义预测和低层动作共同训练来提升开放世界泛化。 |
| VLA 冗余/高效推理 | *EfficientVLA*, arXiv:2506.10100, <https://arxiv.org/abs/2506.10100> | EfficientVLA 直接从多种冗余入手，包括语言层冗余、任务相关视觉 token 选择和扩散动作头的时间冗余。这与本项目“冗余定义”高度相关。 |
| 2026 数据质量过滤趋势 | *Green-VLA*, arXiv:2602.00919, <https://arxiv.org/abs/2602.00919> | Green-VLA 把 temporal alignment 与 quality filtering 放入数据处理管线，说明 VLA 领域已把数据质量过滤视为系统能力。 |

<a id="sec-1-4"></a>

### 1.4 严格限制

- 不训练 OpenVLA、pi0、SmolVLA 等大模型作为主线。本题硬要求是冻结视觉特征 + 轻量 MLP，且课程数据集只有 50 episodes。
- 不把“语言”夸大。该数据集 `total_tasks = 1`，语言指令在主实验中基本是常量，因此语言特征只作为 VLA 输入格式的一部分；报告中必须明确这一限制。
- 不在 test episodes 上训练 selector、MLP、normalizer 或调参。
- 如果最终 PC-RAS-Coreset 没有稳定超过随机 baseline，应如实报告，并通过消融和失败分析解释原因，不能硬编结论。

---

<a id="sec-2"></a>

## 2. 总体课设架构

<a id="sec-2-1"></a>

### 2.1 推荐主线

本项目分为三层：

1. **课程要求层**：随机 10% episode baseline，冻结 CLIP/ResNet-18 特征，MLP 预测单臂 7D action，MSE 评估。
2. **方法创新层**：PC-RAS-Coreset，按脑启发机制定义数据价值，从候选训练池选择 10% 高价值 episode。
3. **研究加分层**：帧级 10% coreset、预算曲线、冗余可视化、phase-aware MSE、selector 消融。

<a id="sec-2-2"></a>

### 2.2 主实验选择单位

题面写的是“随机抽取 10% 的轨迹样本”，因此主评测必须以 episode 为单位：

- 数据集总共 50 episodes。
- 10% 轨迹样本 = 5 episodes。
- 随机 baseline：从训练候选池随机选 5 episodes。
- PC-RAS-Coreset：从训练候选池自动选 5 episodes。
- 两者使用完全相同 MLP、特征、训练超参、验证集和测试集。

扩展实验再做帧级选择：

- 帧级 10% = 2000 frames。
- 目的：更直接验证“时序冗余过滤”。
- 报告中标为 extra，不替代主实验。

<a id="sec-2-3"></a>

### 2.3 推荐数据划分

为了避免时间相邻帧泄漏，所有划分都按 episode 做：

- `candidate_train`: episodes 0-39，共 40 条轨迹。selector 只能从这里选。
- `val`: episodes 40-44，共 5 条轨迹。只用于 early stopping 和模型选择。
- `test`: episodes 45-49，共 5 条轨迹。只用于最终报告 MSE。

主实验训练集：

- Random-10%：从 episodes 0-39 中随机选 5 episodes。
- PC-RAS-10%：从 episodes 0-39 中自动选 5 episodes。

为抵消小数据集波动：

- Random-10% 至少跑 5 个 seed：`0, 1, 2, 3, 4`。
- PC-RAS-10% 至少跑 3 个 seed：差异来自 MLP 初始化；selector 若有随机 PCA/kmeans，也固定 seed。
- 报告均值、标准差、最好/最差 seed。

---

<a id="sec-3"></a>

## 3. 数据字段与任务定义

<a id="sec-3-1"></a>

### 3.1 输入字段

主实验使用：

- 图像：`observation.images.top`
- 语言：常量指令，建议设为 `"transfer the cube"` 或从 `dataset.meta` 中读取任务文本后固定使用。
- 标签：`action[:7]`，即左臂 7D action。

为什么选左臂：

- 官方 metadata 显示 `action` 维度为 14，电机名称顺序是左臂 7 维 + 右臂 7 维。
- 课程题目允许“仅提取其中一个视角的画面和单臂动作标签进行降维实验”。
- 左臂维度定义：
  - `left_waist`
  - `left_shoulder`
  - `left_elbow`
  - `left_forearm_roll`
  - `left_wrist_angle`
  - `left_wrist_rotate`
  - `left_gripper`

可选附加实验：

- 右臂 7D：`action[7:14]`
- 双臂 14D：只作为扩展，不作为主报告指标。
- 加入 proprioception：`observation.state[:7]`。这会偏离题面 `[视觉+语言] -> action`，只能作为 ablation。

<a id="sec-3-2"></a>

### 3.2 输出指标

主指标：

- `MSE_7D`: 对 test set 所有帧、所有 7 个动作维度求均方误差。

必须同时报告：

- 每个动作维度的 MSE。
- `gripper` 维度 MSE。
- 按 phase 分组的 MSE：
  - low-motion phase
  - high-motion phase
  - gripper-transition phase

训练时可以对 action 做标准化，但报告 MSE 应同时给：

- normalized-space MSE：便于模型训练对比。
- original-action-space MSE：便于物理意义解释。

---

<a id="sec-4"></a>

## 4. 特征与模型选择

<a id="sec-4-1"></a>

### 4.1 推荐特征提取器

主特征提取器：

- `open_clip` 或 `transformers` CLIP ViT-B/32
- 图像 embedding：512D
- 文本 embedding：512D
- 输入给 MLP：`[image_emb, text_emb]`，共 1024D

选择 CLIP 的原因：

- 题目明确允许 CLIP。
- CLIP 是图文共同训练模型，能自然满足 `[视觉特征 + 语言指令]` 的形式。
- 虽然本数据集只有 1 个 task，语言 embedding 是常量，但这种设计便于报告中说明 VLA-lite 形式和局限。

备用/消融特征：

- torchvision ResNet-18 ImageNet-1K 预训练模型
- 使用去掉分类头后的 512D feature
- 拼接同一个 CLIP text embedding 或一个固定 task embedding，保证输入格式一致

<a id="sec-4-2"></a>

### 4.2 MLP 主模型

推荐结构：

```text
input_dim = 1024  # CLIP image 512 + text 512
Linear(input_dim, 512)
LayerNorm(512)
GELU
Dropout(0.1)
Linear(512, 256)
LayerNorm(256)
GELU
Dropout(0.1)
Linear(256, 7)
```

训练设置：

- loss：MSELoss
- optimizer：AdamW
- lr：`1e-3`
- weight_decay：`1e-4`
- batch_size：`256` 或 `512`
- max_epochs：`200`
- early_stopping_patience：`20`
- seed：至少 `0,1,2,3,4`
- action normalization：用当前训练子集统计量拟合 mean/std，val/test 只 transform，不参与拟合

计算资源：

- 两张 48G RTX 4090 远超主实验需求。
- 推荐 GPU0 做 CLIP/ResNet feature extraction，GPU1 并行跑不同 seed 的 MLP。
- 由于数据只有 20k 帧，MLP 训练一般是分钟级。

---

<a id="sec-5"></a>

## 5. PC-RAS-Coreset 方法设计

<a id="sec-5-1"></a>

### 5.1 冗余定义

本项目将 VLA 数据冗余拆成三类：

1. **时序冗余**：连续帧视觉变化很小、动作变化很小，对学习动作映射贡献低。
2. **分布冗余**：大量样本位于相似视觉/动作区域，重复覆盖同一状态。
3. **低效/次优噪声**：人类遥操作中的停顿、犹豫、无效微调；在本数据集没有显式失败标签，因此只能用低运动 + 低视觉变化 + 低任务进展代理识别，不能声称一定是错误示范。

<a id="sec-5-2"></a>

### 5.2 脑启发映射

| 脑机制 | 工程化指标 | 本项目含义 |
| --- | --- | --- |
| 预测编码 | temporal surprise / prediction error | 只有当图像或动作打破上一时刻预测时，样本才更有价值。 |
| RAS 任务过滤 | task utility score | 机械爪接近开合、动作突变、任务阶段转换等时刻更值得保留。 |
| 注意与覆盖 | diversity / k-center coverage | 不只选最高分帧，还要覆盖不同阶段和动作分布，避免 coreset 偏科。 |

<a id="sec-5-3"></a>

### 5.3 帧级分数

对每个候选训练帧 `t` 计算以下分数。

#### 5.3.1 视觉时序惊奇度

设 CLIP image feature 为 `z_t`，先做 L2 normalize，再可选 PCA 到 64D。

```text
visual_delta_t = ||z_t - z_{t-1}||_2
```

更强版本：

```text
pred_z_t = Ridge([z_{t-1}, a_{t-1}, phase_{t-1}])
visual_prediction_error_t = ||z_t - pred_z_t||_2
```

MVP 推荐先用 `visual_delta_t`，扩展实验再加 Ridge residual。

#### 5.3.2 动作惊奇度

设单臂动作为 `a_t in R^7`。

一阶动作变化：

```text
action_delta_t = ||a_t - a_{t-1}||_2
```

二阶动作 jerk：

```text
action_jerk_t = ||a_t - 2*a_{t-1} + a_{t-2}||_2
```

局部动作方差：

```text
action_var_t = mean_var(a_{t-w:t+w}), w = 3 or 5
```

推荐组合：

```text
pc_score_t = robust_norm(visual_delta_t)
           + 0.7 * robust_norm(action_delta_t)
           + 0.5 * robust_norm(action_jerk_t)
```

`robust_norm(x)` 使用训练候选池上的 median/IQR 或 percentile min-max，避免极端值支配。

#### 5.3.3 RAS 任务效用分数

Transfer Cube 的关键阶段通常与机械爪和动作轨迹突变相关。由于本数据集没有显式接触标签，本项目用代理指标：

```text
gripper_t = a_t[6]
gripper_delta_t = |gripper_t - gripper_{t-1}|
motion_norm_t = ||a_t - a_{t-1}||_2
phase_t = frame_index_t / (episode_length - 1)
```

任务效用：

```text
ras_score_t = 1.0 * robust_norm(gripper_delta_t)
            + 0.7 * robust_norm(motion_norm_t)
            + 0.3 * phase_balance_bonus_t
```

`phase_balance_bonus_t` 的作用不是让某个阶段得分最高，而是避免 10% 样本全挤在同一阶段。实现方式：

- 把每个 episode 分成 5 个 phase bin：`[0, .2)`, `[.2, .4)`, `[.4, .6)`, `[.6, .8)`, `[.8, 1.0]`
- selector 对每个 phase bin 设置最小覆盖约束
- 如果某个 phase bin 当前覆盖不足，则给该 bin 候选样本小额 bonus

#### 5.3.4 分布覆盖分数

构建 coreset embedding：

```text
e_t = concat(
  PCA64(CLIP_image_feature_t),
  normalized_action_t,
  phase_t
)
```

使用 k-center greedy 或 k-means medoid 做覆盖：

- k-center：每次选择离当前 selected set 最远、同时 PC/RAS 分数较高的样本。
- k-means medoid：先按 `e_t` 聚类，再从每个 cluster 选高 PC/RAS 分数样本。

推荐主实现：

```text
final_frame_score_t = 0.45 * pc_score_t
                    + 0.35 * ras_score_t
                    + 0.20 * coverage_score_t
```

其中 `coverage_score_t` 在 greedy 过程中动态更新：

```text
coverage_score_t = min_distance(e_t, selected_embeddings)
```

<a id="sec-5-4"></a>

### 5.4 episode 级选择

主评测要选 5 条 episode。先把帧级分数聚合成 episode 分数：

```text
episode_saliency_i = mean(top 20% final_frame_score within episode i)
episode_motion_i = mean(action_delta within episode i)
episode_diversity_emb_i = mean(e_t within episode i)
```

然后进行 episode 级 diversity-aware greedy：

```text
selected = []
while len(selected) < 5:
    choose episode i maximizing:
        0.65 * robust_norm(episode_saliency_i)
      + 0.35 * min_distance(episode_diversity_emb_i, selected_episode_embeddings)
      + phase_coverage_bonus_i
```

这能避免只选高动作幅度 episode，保留不同轨迹风格。

<a id="sec-5-5"></a>

### 5.5 帧级 10% 扩展

如果做帧级 10%：

- 从 `candidate_train` 的 16000 帧中选 2000 帧。
- 为避免相邻帧全被选中，加入 temporal non-maximum suppression：

```text
same episode 内，若两个候选帧距离 < 5 frames，只保留 final_frame_score 更高者
```

训练 MLP 时直接用选中的 2000 帧。

---

<a id="sec-6"></a>

## 6. 实验矩阵

<a id="sec-6-1"></a>

### 6.1 必做实验

| 实验 | 训练数据 | 特征 | 模型 | 目的 |
| --- | --- | --- | --- | --- |
| Full upper bound | episodes 0-39 全部 | CLIP | MLP | 给 10% 方法一个上限参考，不作为题面主对比。 |
| Random-10% | 随机 5 episodes | CLIP | 同一个 MLP | 题面 baseline。 |
| PC-RAS-10% | 自动选 5 episodes | CLIP | 同一个 MLP | 主方法。 |
| Random-10% + ResNet18 | 随机 5 episodes | ResNet18 | 同一个 MLP | 验证结论不完全依赖 CLIP。 |
| PC-RAS-10% + ResNet18 | 自动选 5 episodes | ResNet18 | 同一个 MLP | encoder robustness。 |

<a id="sec-6-2"></a>

### 6.2 方法消融

| 实验 | 选择规则 | 目的 |
| --- | --- | --- |
| ActionVar-only | 只按动作变化/方差选 episode | 验证题面给的简单思路。 |
| VisualDelta-only | 只按视觉变化选 episode | 验证预测编码视觉部分。 |
| Coverage-only | k-center/k-means coverage | 验证分布覆盖。 |
| PC-only | visual/action surprise | 验证预测编码组合。 |
| PC+RAS | 不加 coverage | 看高效用是否会过度集中。 |
| PC+RAS+Coverage | 完整方法 | 主方法。 |

<a id="sec-6-3"></a>

### 6.3 额外研究实验

| 实验 | 目的 |
| --- | --- |
| Budget curve: 2%, 5%, 10%, 20% | 证明不是只在 10% 点偶然有效。 |
| Frame-level 10% | 更直接验证时序冗余过滤。 |
| Phase-aware MSE | 分析方法是否改善接触/开合/动作突变阶段。 |
| Selected-frame visualization | 展示 PC/RAS 选择的帧是否确实对应关键事件。 |
| Robustness over random seeds | 用均值和标准差减少小数据误判。 |

---

<a id="sec-7"></a>

## 7. 推荐代码目录结构

在项目根目录下建议采用以下结构：

```text
Project_jfw/
  2026春季认知工程考查试卷.pdf
  document/
    topic2_vla_coreset_project_guide.md
  configs/
    base_clip.yaml
    base_resnet18.yaml
    coreset_pc_ras.yaml
  data/
    raw/
      README.md
    cache/
      lerobot/
    splits/
      split_v1.json
    features/
      clip_vit_b32_top_left7/
      resnet18_top_left7/
    coresets/
      random_seed0.json
      pc_ras_episode_top5.json
      pc_ras_frame_2000.json
  src/
    vla_coreset/
      __init__.py
      data/
        lerobot_loader.py
        split.py
        schema.py
      features/
        extract_clip.py
        extract_resnet18.py
        text_features.py
      selection/
        scores.py
        episode_selector.py
        frame_selector.py
        coverage.py
        baselines.py
      models/
        mlp.py
      training/
        train_mlp.py
        losses.py
      evaluation/
        metrics.py
        phase_metrics.py
      visualization/
        plot_scores.py
        plot_embeddings.py
        plot_predictions.py
        make_report_figures.py
      utils/
        seed.py
        io.py
        logging.py
  scripts/
    00_check_env.py
    01_inspect_dataset.py
    02_make_splits.py
    03_extract_features.py
    04_select_coreset.py
    05_train_mlp.py
    06_evaluate.py
    07_make_figures.py
  experiments/
    runs/
  results/
    tables/
    figures/
    predictions/
  reports/
    report_outline.md
    final_report_assets/
  tests/
    test_scores.py
    test_splits.py
    test_metrics.py
```

原则：

- `data/cache/` 和 `data/features/` 不提交大文件。
- `data/splits/*.json`、`data/coresets/*.json`、`results/tables/*.csv` 可提交，便于复现。
- 所有实验配置写入 YAML，避免报告结果不可追踪。

---

<a id="sec-8"></a>

## 8. 逐步执行流程

<a id="sec-8-0"></a>

### Step 0：环境准备

本项目实验环境统一使用新环境 `coredataset`。`file_env` 仅用于文档读写，不用于本项目的数据集加载、特征提取、训练和评估。

如果找不到 `coredataset` 或无法 activate，先执行：

```bash
source /home/xiaobo.xia/JiafengWu/env.sh
```

然后：

```bash
conda activate coredataset
```

如果后续在终端中访问 Hugging Face、arXiv、Google Scholar 或下载依赖时外网不可达，先加载代理环境：

```bash
JIAFENG_USE_PROXY=1 source /home/xiaobo.xia/JiafengWu/env.sh
conda activate coredataset
```

建议依赖：

```bash
pip install torch torchvision torchaudio
pip install "lerobot>=0.4.0" datasets transformers open_clip_torch
pip install scikit-learn pandas pyarrow numpy matplotlib seaborn tqdm rich pyyaml
```

如果 pip 上的 `lerobot` 版本不含 v3 dataset 支持，使用 Hugging Face 官方仓库安装：

```bash
pip install git+https://github.com/huggingface/lerobot.git
```

注意：

- 只有实际执行下载或安装时才需要网络。
- 本指导文档不要求现在安装。

<a id="sec-8-1"></a>

### Step 1：数据集检查

目标：

- 确认能加载 `lerobot/aloha_sim_transfer_cube_human`。
- 打印每个字段 shape。
- 随机保存 8 张 top camera 图像用于报告。
- 统计每个 episode 帧数是否为 400。
- 检查 action 前 7 维和后 7 维的范围。

输出：

```text
results/tables/dataset_summary.csv
results/figures/dataset_samples.png
```

必须记录：

- `total_episodes`
- `total_frames`
- `fps`
- image key
- action dim
- state dim
- task count

<a id="sec-8-2"></a>

### Step 2：固定划分

生成：

```text
data/splits/split_v1.json
```

推荐内容：

```json
{
  "candidate_train": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
  "val": [40, 41, 42, 43, 44],
  "test": [45, 46, 47, 48, 49],
  "notes": "Episode-level split. Selectors can only use candidate_train."
}
```

注意：

- 所有方法共用同一个 split。
- 不能让 selector 根据 test MSE 反复改权重。

<a id="sec-8-3"></a>

### Step 3：离线特征提取

CLIP：

```text
input: observation.images.top
text: "transfer the cube"
output:
  data/features/clip_vit_b32_top_left7/features.npy
  data/features/clip_vit_b32_top_left7/index.parquet
  data/features/clip_vit_b32_top_left7/actions_left7.npy
  data/features/clip_vit_b32_top_left7/text_features.npy
```

ResNet18：

```text
data/features/resnet18_top_left7/features.npy
```

`index.parquet` 至少包含：

- global row index
- episode_index
- frame_index
- timestamp
- split
- action_left_0 到 action_left_6

<a id="sec-8-4"></a>

### Step 4：随机 baseline 选择

对每个 seed：

```text
randomly choose 5 episodes from candidate_train
save to data/coresets/random_episode_seed{seed}.json
```

训练 MLP 并评估：

```text
results/tables/random10_clip_seed{seed}_metrics.csv
results/predictions/random10_clip_seed{seed}.npz
```

最终汇总：

```text
results/tables/random10_clip_summary.csv
```

<a id="sec-8-5"></a>

### Step 5：计算 PC-RAS 分数

为每个 candidate frame 计算：

- visual_delta
- action_delta
- action_jerk
- gripper_delta
- phase
- pc_score
- ras_score
- coverage embedding
- final_frame_score

保存：

```text
data/coresets/frame_scores_clip.parquet
results/figures/score_timeline_episode_*.png
```

检查点：

- 高分帧是否集中在动作变化大、机械爪开合、任务阶段变化的位置。
- 如果高分全在同一段，调低 RAS 权重或加强 phase coverage。

<a id="sec-8-6"></a>

### Step 6：episode 级 PC-RAS-Coreset

从 episodes 0-39 中选 5 个 episodes：

```text
data/coresets/pc_ras_episode_top5.json
```

同时输出解释：

```text
results/tables/pc_ras_selected_episode_explanations.csv
```

每个 episode 的解释字段：

- episode_index
- episode_saliency
- top_frame_indices
- phase coverage
- mean visual_delta
- mean action_delta
- mean gripper_delta
- nearest selected distance
- selected reason

<a id="sec-8-7"></a>

### Step 7：训练 PC-RAS MLP

用同样 MLP 和同样训练脚本：

```text
train episodes = selected 5 episodes
val episodes = 40-44
test episodes = 45-49
```

输出：

```text
results/tables/pc_ras10_clip_seed{seed}_metrics.csv
results/predictions/pc_ras10_clip_seed{seed}.npz
```

<a id="sec-8-8"></a>

### Step 8：消融实验

至少跑：

- ActionVar-only
- VisualDelta-only
- Coverage-only
- PC-only
- PC+RAS
- PC+RAS+Coverage

输出统一表：

```text
results/tables/ablation_summary.csv
```

<a id="sec-8-9"></a>

### Step 9：可视化

必须生成以下图：

1. 数据样例图：top camera 图像网格。
2. 随机 vs PC-RAS 选中 episode 的 action 曲线。
3. 每个 episode 的 score timeline。
4. PCA/t-SNE/UMAP 覆盖图：全数据、random 10%、PC-RAS 10%。
5. MSE 对比柱状图：mean ± std。
6. per-dimension MSE heatmap。
7. phase-aware MSE bar chart。
8. 选中关键帧可视化：每张图标注 `PC score / RAS score / phase / reason`。

<a id="sec-8-10"></a>

### Step 10：报告撰写

报告建议结构：

```text
1. 引言：VLA 数据冗余与算力墙
2. 背景调研：ACT、data pruning、predictive coding、OpenVLA/SmolVLA/EfficientVLA/Green-VLA
3. 数据集与任务定义：ALOHA Sim Transfer Cube，单臂 7D action，VLA-lite 限制
4. Baseline：随机 10% episode + CLIP/ResNet18 + MLP
5. 方法：PC-RAS-Coreset
6. 实验设置：split、metrics、seeds、模型超参
7. 结果：主对比、消融、预算曲线、phase-aware MSE
8. 认知工程分析：预测编码、RAS、注意覆盖分别对应什么失败/改进
9. 局限性：单任务语言常量、小数据集、MSE 不等价于闭环成功率
10. 结论
```

---

<a id="sec-9"></a>

## 9. 报告中的核心论点

<a id="sec-9-1"></a>

### 9.1 推荐论点

本项目要把“10% 数据更好”说清楚，不能只说分数提升。推荐论点如下：

1. 随机 10% episode 容易抽到大量低运动或重复阶段，训练集覆盖不足。
2. 预测编码分数能保留视觉/动作发生显著变化的关键时刻。
3. RAS 任务效用分数能提高机械爪开合、动作突变、阶段转换附近样本的权重。
4. 单纯高分选择会偏向少数阶段，因此必须加入 coverage/phase balance。
5. 完整 PC-RAS-Coreset 如果优于 Random-10%，说明在极小数据预算下，样本质量和分布覆盖比简单数量更关键。

<a id="sec-9-2"></a>

### 9.2 独特价值

本项目的独特价值不在于发明复杂模型，而在于：

- 把认知机制落到可计算指标，而不是停留在文字类比。
- 对“冗余”给出三层定义：时序冗余、分布冗余、低效噪声。
- 用可视化解释 selector 为什么选这些 episode/frame。
- 在只有 50 episodes 的轻量数据集上做严谨的 seed、split、ablation。
- 与 2025/2026 VLA 的效率趋势接轨：数据质量过滤、token/时间冗余削减、小模型部署。

---

<a id="sec-10"></a>

## 10. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 数据集只有 1 个 task，语言 embedding 是常量 | 语言对 MSE 几乎无贡献 | 报告中明确称为 VLA-lite；保留语言输入形式，但不夸大语言 grounding。 |
| 5 episode 训练集太小，随机波动大 | 单次结果不可靠 | 多 seed，报告 mean/std；用 fixed test set；补充 budget curve。 |
| PC/RAS 过度选择高动态帧 | 低运动阶段 MSE 变差 | episode 级选择用完整轨迹；frame 级选择加 phase balance。 |
| MSE 改善不明显 | 主结论变弱 | 做 per-phase MSE，可能关键阶段改善但整体平均不大；如仍无改善，如实分析。 |
| selector 权重被 test set 调参污染 | 结果不可用 | 只用 val 调权重；test 最后一次评估。 |
| LeRobot API 版本变化 | 代码加载失败 | 在 `scripts/01_inspect_dataset.py` 中打印版本和 schema；必要时用 HF datasets/parquet 直接读取。 |

---

<a id="sec-11"></a>

## 11. 推荐配置草案

`configs/base_clip.yaml`：

```yaml
project:
  name: pc_ras_vla_aloha
  seed: 0

dataset:
  repo_id: lerobot/aloha_sim_transfer_cube_human
  image_key: observation.images.top
  action_key: action
  arm: left
  action_slice: [0, 7]
  text_instruction: transfer the cube
  split_file: data/splits/split_v1.json

features:
  encoder: clip
  model_name: ViT-B-32
  pretrained: openai
  batch_size: 128
  pca_dim_for_selection: 64

model:
  input_dim: 1024
  hidden_dims: [512, 256]
  dropout: 0.1
  output_dim: 7

training:
  batch_size: 256
  max_epochs: 200
  patience: 20
  lr: 0.001
  weight_decay: 0.0001
  normalize_actions: true

selection:
  budget_episodes: 5
  weights:
    pc: 0.45
    ras: 0.35
    coverage: 0.20
  phase_bins: 5
  top_frame_percent_for_episode_score: 0.20

evaluation:
  metrics: [mse, per_dim_mse, phase_mse]
```

---

<a id="sec-12"></a>

## 12. 最小可交付版本与增强版本

<a id="sec-12-1"></a>

### 12.1 最小可交付版本

必须完成：

- 数据集加载与检查。
- 固定 split。
- CLIP feature extraction。
- Random-10% episode baseline，5 seeds。
- PC-RAS episode selector。
- PC-RAS-10% 训练与测试。
- MSE、per-dim MSE。
- 至少 4 类图：样例、选择可视化、MSE 对比、动作曲线。
- 报告完整写清楚方法和结果。

<a id="sec-12-2"></a>

### 12.2 高质量增强版本

建议完成：

- ResNet18 encoder 对照。
- ActionVar/VisualDelta/Coverage 消融。
- budget curve。
- frame-level 10% selector。
- phase-aware MSE。
- 选中帧解释表。
- 失败案例：PC-RAS 什么时候不如 random。

---

<a id="sec-13"></a>

## 13. 预期结果格式

主表：

| Method | Encoder | Budget | Unit | Test MSE ↓ | Gripper MSE ↓ | High-motion MSE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Full upper bound | CLIP | 100% train pool | episode | 实验产出填入 | 实验产出填入 | 实验产出填入 |
| Random-10% | CLIP | 5 episodes | episode | mean ± std | mean ± std | mean ± std |
| PC-RAS-10% | CLIP | 5 episodes | episode | mean ± std | mean ± std | mean ± std |
| Random-10% | ResNet18 | 5 episodes | episode | mean ± std | mean ± std | mean ± std |
| PC-RAS-10% | ResNet18 | 5 episodes | episode | mean ± std | mean ± std | mean ± std |

消融表：

| Selector | PC | RAS | Coverage | Test MSE ↓ | 解释 |
| --- | --- | --- | --- | --- | --- |
| Random | no | no | no | 实验产出填入 | 随机轨迹 baseline |
| ActionVar-only | partial | no | no | 实验产出填入 | 只看动作变化 |
| VisualDelta-only | partial | no | no | 实验产出填入 | 只看视觉变化 |
| Coverage-only | no | no | yes | 实验产出填入 | 覆盖状态分布 |
| PC+RAS | yes | yes | no | 实验产出填入 | 可能偏关键阶段 |
| PC+RAS+Coverage | yes | yes | yes | 实验产出填入 | 完整方法 |

---

<a id="sec-14"></a>

## 14. 工作优先级

推荐按以下顺序推进：

1. 先实现数据加载、split 和 CLIP feature cache。
2. 跑通 Random-10% baseline，确认 MSE pipeline 正确。
3. 实现最简单 ActionVar-only selector，作为 sanity check。
4. 实现 PC-RAS 完整 selector。
5. 做主对比和图。
6. 再做 ResNet18、消融和预算曲线。
7. 最后写报告。

不要一开始尝试训练大 VLA 或复杂 transformer policy。题目的评分重点是：核心集选择思想、冗余定义、实验对比和认知分析。

---

<a id="sec-15"></a>

## 15. 参考链接清单

- 课程 PDF：`/home/xiaobo.xia/JiafengWu/code_folder/area1/Project_jfw/2026春季认知工程考查试卷.pdf`
- ALOHA Sim Transfer Cube dataset: <https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human>
- LeRobotDataset v3 docs: <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- ACT: <https://arxiv.org/abs/2304.13705>
- Data pruning: <https://arxiv.org/abs/2206.14486>
- Core-set active learning: <https://arxiv.org/abs/1708.00489>
- Predictive coding review: <https://arxiv.org/abs/2107.12979>
- CLIP: <https://arxiv.org/abs/2103.00020>
- PyTorch ResNet18 docs: <https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html>
- OpenVLA: <https://arxiv.org/abs/2406.09246>
- SmolVLA: <https://arxiv.org/abs/2506.01844>
- pi0: <https://arxiv.org/abs/2410.24164>
- pi0.5: <https://arxiv.org/abs/2504.16054>
- EfficientVLA: <https://arxiv.org/abs/2506.10100>
- Green-VLA: <https://arxiv.org/abs/2602.00919>
