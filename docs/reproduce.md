# 复现说明

本仓库保留竞赛提交的独立脚本，运行前需要准备官方赛题附件。最终论文的费用数字对应当时选定的交付版本；下列命令提供复算入口，不保证在不同 Python / 库版本下逐位相同。

## 1. 环境与输入

建议使用 Python 3.13。提交版本的一份留档环境为 Python 3.13.7，NumPy 2.5.3、Pandas 3.0.5、SciPy 1.18.1、scikit-learn 1.9.1、LightGBM 4.7.0、Numba 0.67.0、openpyxl 3.1.5。安装本仓库列出的依赖：

```bash
python -m pip install -r requirements.txt
```

从[竞赛官网](https://www.mcm.edu.cn/html_cn/node/27b6e148f8113f09b0269f64a02629fb.html)下载 2026 年赛题包，将其中 C 题附件整理为：

```text
data/
  附件1.xlsx
  附件2.xlsx
  附件3.xlsx
  附件4.xlsx
  附件5/
    result1.xlsx
    result2.xlsx
    result3.xlsx
    result4-2.xlsx
    result4-3.xlsx
```

保留 `附件5` 的**原始空白模板**，不要将已填的交付结果当作模板。脚本均通过命令行接收输入路径，不读取仓库外的旧项目缓存。以下命令在仓库根目录运行。

## 2. 问题一：确定性 LP / MILP

```bash
mkdir outputs/q1
python code/q1_complete.py --input data/附件1.xlsx --template data/附件5/result1.xlsx --output outputs/q1/result1.xlsx
```

输出包括 `result1.xlsx`、十分钟调度明细和时间映射。脚本会校验模板，并阻止输出覆盖原始输入。

## 3. 问题二：预测与储能反馈

```bash
python code/q2_complete.py --attachment1 data/附件1.xlsx --attachment2 data/附件2.xlsx --template data/附件5/result2.xlsx --mode archive --archive artifacts/base_forecasts_submission.npz --output-dir outputs/q2_archive --workers 1 --threads 1
```

`archive` 读取提交版本留档的四分量基础预测，仍重新执行融合、购电规划和回放。若要从官方附件重新训练预测器，将 `--mode archive --archive ...` 改为 `--mode retrain`，并另设输出目录。重训费用可与论文交付值不同；不能将留档复算标为“本次完整重训”。

## 4. 问题三：多时刻滚动改约

```bash
python code/q3_complete.py --attachment1 data/附件1.xlsx --attachment2 data/附件2.xlsx --attachment3 data/附件3.xlsx --template data/附件5/result3.xlsx --mode archive --archive artifacts/base_forecasts_submission.npz --policy main --output-dir outputs/q3_archive --workers 1 --threads 1
```

脚本另有 `--policy baseline` 和 `--policy both`，用于同一问题内的基线比较。不同策略应使用独立输出目录，避免混用运行来源记录。

## 5. 问题四：波动电价两类合同

```bash
python code/q4_2_complete.py --attachment2 data/附件2.xlsx --attachment3 data/附件3.xlsx --attachment4 data/附件4.xlsx --template data/附件5/result4-2.xlsx --mode archive --archive artifacts/base_forecasts_submission.npz --output-dir outputs/q4_2_archive --workers 1 --threads 1
python code/q4_3_complete.py --attachment2 data/附件2.xlsx --attachment3 data/附件3.xlsx --attachment4 data/附件4.xlsx --template data/附件5/result4-3.xlsx --mode archive --archive artifacts/base_forecasts_submission.npz --output-dir outputs/q4_3_archive --workers 1 --threads 1
```

4-2 与 4-3 共用大部分求解框架，但交易权限不同，因此保留提交时的两个单文件脚本。每个脚本会生成费用、预测和物理约束的核验记录。

## 复算时的核对顺序

1. 检查脚本生成的 `运行来源与环境.json`：输入文件哈希、代码哈希、Python / 依赖版本及 `run_mode`。
2. 核对输出表、费用摘要与逐区间的供需和储能约束记录。
3. 区分 `archive` 与 `retrain`，只在相同数据、模板、环境和策略口径下比较费用。
4. 将任何费用差异与论文中的同年度选型限制一并报告，不把回放差额直接解释为独立样本优势。
