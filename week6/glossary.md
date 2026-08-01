# Truck-Drone EVRP-TW 项目术语表 (Glossary)

> 收录项目中所有简称、缩写与专有名词，按类别分组，方便查阅与论文撰写。

---

## 1. 问题与模型

| 术语 | 全称 | 解释 |
|------|------|------|
| **VRP** | Vehicle Routing Problem | 车辆路径问题：将货物从仓库配送到一组客户的最优路径规划 |
| **VRPTW** | Vehicle Routing Problem with Time Windows | 带时间窗的VRP：每个客户有最早/最晚服务时间限制 |
| **EVRP** | Electric Vehicle Routing Problem | 电动车路径问题：考虑电池容量与充电站约束 |
| **EVRP-TW** | Electric Vehicle Routing Problem with Time Windows | 带时间窗的电动车路径问题（本项目核心问题） |
| **TSP** | Traveling Salesman Problem | 旅行商问题：VRP的单车辆特例 |
| **FSTSP** | Flying Sidekick Traveling Salesman Problem | 飞行副手TSP：Murray & Chu (2015)提出的卡车-无人机协同模型 |
| **CVRP** | Capacitated Vehicle Routing Problem | 带容量约束的VRP |

---

## 2. Solomon 实例体系

| 术语 | 全称 | 解释 |
|------|------|------|
| **RC1** | Random-Clustered Type 1 | 混合分布 + 紧时间窗 (horizon=120min)，8个实例 RC101-RC108 |
| **RC2** | Random-Clustered Type 2 | 混合分布 + 宽时间窗 (horizon=240min)，8个实例 RC201-RC208 |
| **R1** | Random Type 1 | 随机分布 + 紧时间窗，12个实例 R101-R112 |
| **R2** | Random Type 2 | 随机分布 + 宽时间窗，11个实例 R201-R211 |
| **C1** | Clustered Type 1 | 聚集分布 + 紧时间窗，9个实例 C101-C109 |
| **C2** | Clustered Type 2 | 聚集分布 + 宽时间窗，8个实例 C201-C208 |
| **TW** | Time Window | 时间窗：客户可被服务的时间区间 `[ready_time, due_time]` |
| **TW Horizon** | Time Window Horizon | 时间窗跨度：最早就绪到最晚截止的时间范围（C1/R1/RC1=120min, C2/R2/RC2=240min) |
| **BKS** | Best Known Solution | 已知最优解：Solomon基准库中文献报道的最优目标值 |

---

## 3. 算法与方法

| 术语 | 全称 | 解释 |
|------|------|------|
| **POMO** | Policy Optimization with Multiple Optima | Kwon et al. (2020)提出的神经组合优化方法，利用Transformer从多个最优起点学习路由策略 |
| **EDD** | Earliest Due Date | 最早截止日期优先：经典调度启发式，按due_time升序排列客户，对最小化最大延迟是可证明最优的 |
| **ALNS** | Adaptive Large Neighborhood Search | 自适应大邻域搜索：Ropke & Pisinger (2006)，通过自适应选择破坏/修复算子迭代优化解 |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II | 非支配排序遗传算法：Deb et al. (2002)，多目标进化算法，通过Pareto排序和拥挤度保持种群多样性 |
| **P-ACO** | Pareto Ant Colony Optimization | 帕累托蚁群优化：基于蚁群算法（ACO）的多目标扩展，维护成本+延迟双信息素矩阵 |
| **IVND** | Improved Variable Neighborhood Descent | 改进变邻域下降：7种邻域结构的局部搜索，结合禁忌搜索+模拟退火，来自DOI: 10.1109/TITS.2022.3181282 |
| **VND** | Variable Neighborhood Descent | 变邻域下降：系统切换多种邻域结构进行局部搜索 |
| **SA** | Simulated Annealing | 模拟退火：以概率接受较差解的随机优化方法，温度逐渐降低 |
| **K-means** | K-means Clustering | K均值聚类：将客户按空间坐标分为K个簇 |
| **NN** | Nearest Neighbor | 最近邻启发式：每次选择距离当前位置最近的未访问客户 |
| **2-opt** | 2-opt Exchange | 2-优化交换：删除两条边并重新连接，消除路径交叉 |
| **SBX** | Simulated Binary Crossover | 模拟二进制交叉：遗传算法中用于实数编码的交叉算子 |

---

## 4. 优化概念

| 术语 | 全称 | 解释 |
|------|------|------|
| **MOO** | Multi-Objective Optimization | 多目标优化：同时优化多个相互冲突的目标（本项目：成本 vs 延迟） |
| **Pareto Front** | Pareto Front | 帕累托前沿：非支配解的集合，任一目标的改进必然导致另一目标退化 |
| **HV** | Hypervolume | 超体积：多目标优化质量指标，衡量Pareto前沿支配的参考点空间体积 |
| **Gap** | Optimality Gap | 最优性差距：(启发式解 - 最优解) / 最优解 × 100% |
| **LB** | Lower Bound | 下界：问题最优解的理论下限，用于评估启发式解的质量 |
| **Feasibility** | Feasibility | 可行性：解满足所有约束（容量、时间窗、电池、同步等）的比例 |
| **Tardiness** | Tardiness | 延迟：服务完成时间超出客户due_time的部分，加权计入成本函数 |
| **Makespan** | Makespan | 完工时间：所有车辆完成配送并返回仓库的最晚时间 |
| **Abiation** | Abiation Study | 消融实验：依次移除/添加系统组件，量化每个组件的贡献 |

---

## 5. 无人机与卡车协同

| 术语 | 全称 | 解释 |
|------|------|------|
| **UAV** | Unmanned Aerial Vehicle | 无人机（与Drone同义） |
| **Drone Mission** | Drone Mission | 无人机任务：表示为三元组 `(i, j, k)`，无人机从节点i发射→服务客户j→在节点k被卡车回收 |
| **Launch Node** | Launch Node (i) | 发射节点：无人机从卡车上起飞的客户位置 |
| **Recovery Node** | Recovery Node (k) | 回收节点：无人机降落回卡车的客户位置 |
| **Cross-Route Drone** | Cross-Route Drone Insertion | 跨路线无人机插入：卡车A的无人机服务卡车B路线上的客户 |
| **Endurance** | Drone Endurance | 无人机续航：最大飞行距离（本项目：4km medium / 6km high） |
| **Sync** | Synchronization | 同步：卡车和无人机在发射点和回收点的时间协调 |
| **Sync Violation** | Synchronization Violation | 同步违规：无人机到达回收点时卡车尚未到达（无人机无处降落） |
| **Truck Wait Time** | Truck Waiting Time | 卡车等待时间：卡车在回收点等待无人机到达的空闲时间 |

---

## 6. 电动车相关

| 术语 | 全称 | 解释 |
|------|------|------|
| **EV** | Electric Vehicle | 电动车（本项目特指电动卡车） |
| **SOC** | State of Charge | 电池荷电状态：当前电量占电池容量的百分比 [0, 1] |
| **CS** | Charging Station | 充电站：卡车可在此补充电量的节点 |
| **Linear Charging** | Linear Charging | 线性充电：充电速率恒定，充电时间 = 所需能量 / 充电速率 |
| **Non-linear Charging** | Non-linear Charging | 非线性充电：充电速率随SOC变化（低SOC快充→高SOC慢充），更接近真实电池行为 |
| **Energy per km** | Energy per km | 每公里能耗：卡车行驶1km消耗的电量（本项目：1.0 kWh/km） |
| **Battery Capacity** | Battery Capacity | 电池容量：卡车电池的最大储电量（本项目：100 kWh） |

---

## 7. 实验与评估

| 术语 | 全称 | 解释 |
|------|------|------|
| **Reproducibility** | Reproducibility | 可复现性：固定随机种子、记录所有参数，确保实验结果可被独立复现 |
| **Seed** | Random Seed | 随机种子：用于控制伪随机数生成，确保实验可复现（本项目基准种子=42） |
| **Runtime** | Runtime | 运行时间：算法从开始到结束的墙钟时间（秒） |
| **Pareto Dominance** | Pareto Dominance | 帕累托支配：解A在所有目标上≤解B，且至少在一个目标上严格优于B |
| **Feasibility Rate** | Feasibility Rate | 可行率：多次运行中满足所有约束的解的比例 |
| **Mean Cost** | Mean Cost | 平均成本：多次运行的目标函数平均值 |
| **Std** | Standard Deviation | 标准差：多次运行结果的离散程度 |

---

## 8. Week 6-7 专有概念

| 术语 | 全称 | 解释 |
|------|------|------|
| **Partial EDD** | Partial (Segment-Level) EDD Repair | 局部EDD修复：仅对路线中存在延迟的连续片段进行EDD重排，保留其他路段的POMO距离优化 |
| **Full EDD** | Full-Route EDD Repair | 全局EDD修复：对整条路线进行EDD重排，消除所有延迟但可能破坏距离优化 |
| **Repair Phase Transition** | Repair Phase Transition | 修复相变：≤50客户时Partial EDD更优，100客户时Full EDD更优，临界点约75客户 |
| **Fallback** | Fallback Mechanism | 回退机制：当局部EDD修复未能消除延迟时，自动回退到全局EDD修复 |
| **Meta-Learner** | Meta-Learner (Strategy Selector) | 元学习器：基于实例特征（TW类型、客户数、TW密度等）预测最优聚类策略的决策树模型 |
| **Fine-Tuning** | POMO Fine-Tuning | POMO微调：在Solomon实例上对预训练POMO模型进行额外训练（结论：无效，预训练已到局部最优） |
| **Cluster Feasibility** | Temporal Cluster Feasibility | 簇时间可行性：在路由之前检查每个簇是否理论上可被单辆卡车按时服务 |
| **Adaptive Strategy** | Adaptive Repair Strategy | 自适应修复策略：根据实例规模自动选择Partial EDD（≤50c）或Full EDD（100c） |
| **Hybrid Clustering** | Hybrid Clustering | 混合聚类：RC1实例用角度扇形聚类，RC2实例用自适应TW聚类 |

---

## 9. 技术栈

| 术语 | 全称 | 解释 |
|------|------|------|
| **PyTorch** | PyTorch | 深度学习框架，POMO模型的运行环境 |
| **OR-Tools** | Google OR-Tools | Google运筹优化工具包，本项目用于小实例VRPTW精确求解 |
| **PyVRP** | PyVRP | 基于混合遗传搜索(HGS)的VRP求解器（本项目未使用，仅评估） |
| **CP-SAT** | Constraint Programming - SAT | OR-Tools中的约束规划求解器，用于组合优化问题的精确求解 |
| **GLS** | Guided Local Search | 引导式局部搜索：OR-Tools的元启发式改进策略 |
| **Transformer** | Transformer Architecture | 基于自注意力机制的神经网络架构，POMO的编码器基础 |
| **SBX** | Simulated Binary Crossover | 遗传算法中模拟二进制交叉的实数编码算子 |

---

## 10. 论文与参考文献 DOI

| 简称 | 完整引用 | DOI |
|------|---------|-----|
| **POMO** | Kwon et al. (2020) "POMO: Policy Optimization with Multiple Optima" | — |
| **ALNS** | Ropke & Pisinger (2006) "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem" | — |
| **NSGA-II** | Deb et al. (2002) "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" | — |
| **P-ACO** | Das et al. (2020) "Synchronized Truck and Drone Routing in Package Delivery Logistics" | 10.1109/TITS.2020.2992549 |
| **IVND** | Wu et al. (2022) "Collaborative Truck-Drone Routing for Contactless Parcel Delivery During the Epidemic" | 10.1109/TITS.2022.3181282 |
| **EVRP-TW** | Schneider et al. (2014) "The Electric Vehicle-Routing Problem with Time Windows" | 10.1287/trsc.2013.0490 |
| **FSTSP** | Murray & Chu (2015) "The Flying Sidekick Traveling Salesman Problem" | — |
| **AM** | Kool et al. (2019) "Attention, Learn to Solve Routing Problems!" | 10.48550/arXiv.1803.08475 |

---

*最后更新: 2026-07-21 · 收录 80+ 术语 · 按需持续补充*
