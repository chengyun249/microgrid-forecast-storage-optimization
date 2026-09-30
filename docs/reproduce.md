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

## 2026-09-30 本地核验记录

使用官方附件 1—4、原始空白结果模板及仓库公开脚本，逐个执行问题一、二、三主方案与对照方案、4-2 和 4-3。后五次均采用仓库中的基础预测留档文件和 `archive` 模式，**本次没有重训四个基础预测器**。运行环境为 Python 3.13.12、NumPy 2.1.3、Pandas 3.0.2、SciPy 1.17.1、scikit-learn 1.8.0、LightGBM 4.6.0、Numba 0.65.1，与上文留档环境不同。

| 场景 | 论文交付费用（元） | 本次复算费用（元） | 复算减论文（元） |
| --- | ---: | ---: | ---: |
| 问题一 LP / MILP | 35,126.95 | 35,126.95 | 0.00 |
| 问题二 | 13,475,274.43 | 13,476,351.81 | +1,077.38 |
| 问题三主方案 | 13,016,898.13 | 13,016,450.40 | −447.73 |
| 问题三直接扩展对照 | 13,094,652.12 | 13,094,405.50 | −246.62 |
| 问题四 4-2 | 14,132,392.08 | 14,138,020.26 | +5,628.18 |
| 问题四 4-3 | 13,634,348.55 | 13,631,067.25 | −3,281.30 |

表中的论文费用保留两位小数；差额由表内两位小数相减。各脚本均成功结束，生成结果模板和核验记录；问题二至四的供需平衡与储电递推最大数值残差在浮点舍入量级（约 10⁻¹³—10⁻¹² kWh），跨日储电量衔接检查为零。问题三的合同窗口检查通过，4-2/4-3 的 334 个正式评价日及交易时点检查通过。问题三本次主方案较同次对照少 **77,955.10 元**，方向与论文一致，但数额并非论文的 **77,754.00 元**。

本次只确认：公开脚本与官方输入可运行，并在本机生成满足脚本所检物理/结算约束的输出；它**没有逐位重现论文除问题一以外的交付费用**。环境版本差异是可能因素，尚未做受控实验确定全部差异来源。物理核验也不等于证明策略全局最优或实现严格的十分钟区间开始前不可预知性：问题二至四的逐段反馈按当前区间实现的净需求作动作近似，后续问题的模型结构和部分设置仍有同年度选型限制。
