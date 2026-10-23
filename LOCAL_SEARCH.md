# 稳定映射下的慢速局部搜索

本轮是 `random-disturb` 分支的新实验。模型、损失和评测复用当前 attention-aligned 实现，四个历史核心文件不修改。新增实现位于 `scripts/local_search/`。训练数据、模型权重与结果只存远端。

## 目标与预算

在 worker **4401371（4×H20）** 上，比较在线局部搜索与原固定三组合。模型初始化 seed11，总训练跨度40 epochs，保持原40-epoch cosine学习率计划。使用同一份383,368条训练输入、512条验证集，续算有效177,390条。所有训练超参数沿用 `configs/reproduction.json`，只关闭逐microbatch打印。正式训练不新增确定性算子设置；确定性设置仅用于smoke验证。

共同训练epoch0–8，随后分成固定对照与搜索路径；共享前8epochs既保证起点一致，也节省重复计算。固定对照保持原配置至epoch40。搜索在epoch8、12、16、20各提出一个子代，与父代竞争2epochs；竞争间隙训练胜者。epoch24冻结配置，继续至40。共4轮，不因结果不理想追加搜索。总预算约80个单模型epoch的计算量（包括固定对照、共享前缀、被丢弃分支），另加smoke和评测；不是等训练FLOPs比较。

GPU0：搜索父代及胜者后续训练；GPU1：扰动子代；GPU2：固定对照；GPU3：文件锁串行评测。前三张卡分别运行独立单卡模型，不做多卡数据并行。

## 配置表示与扰动

程序存储 `[k,q]`，其中CoT省略比例为 `q/6`，初值 `[[3,2],[4,2],[5,4]]`，即 `(3,1/3)、(4,1/3)、(5,2/3)`。对于保留CoT长度m，实际省略步数为 `j=min(m-1,max(1,ceil(m*q/6)))`，通过整数运算避免浮点取整歧义。

每轮只改一个组合的一个坐标±1。要求 `1<=k<=5`、`1<=q<=5`；三个k严格递增，q非递减。实际训练集上截点分布完全不变的比例扰动被过滤。先在有合法候选的扰动类型（k、q）中等概率选一类，再在该类有效邻居中等概率选一个。独立搜索seed为20261015，轮次种子为 `seed+1000003*round`，不消耗模型RNG。

一个训练窗口内配置稳定，每个optimizer step仍用原 `random.Random(11+1000003*step)` 等概率选三个组合之一，同一梯度累积步共用组合。辅助权重0.3，clone cache不detach，无起始步骤提示，无额外decoder，无辅助forward RNG隔离。

## 竞争与择优

父子从完全相同的模型、optimizer、scheduler、RNG、global step开始，使用相同数据顺序和训练步数。例如第0轮比较epoch9、10的验证结果。两个验证准确率取均值，子代至少提高1个百分点才替换父代，否则保留父代。512题×两个时间点意味着子代累计至少净增11个正确答案。该门槛是保守工程规则，不是显著性检验；两个时间点并非独立样本。

胜者的配置和完整训练状态一起继承。不能比较“训练后的子代”和“训练前的父代”，也不以不同长度目标的续算CE决定优胜。每轮保存候选、实际受影响样本数、两次逐题预测、起点审计、决策及胜者checkpoint。所有阶段均可恢复；每阶段只保留最新checkpoint，验证预测与决策长期保留。

## 数据与验证

复用历史 `continuation-dual-random-20261010/dataset.pt` 中所有合法后缀，只改变选取规则。启动前逐项比对全部主训练字段，逐条重建显式CoT后缀，核对原三组合目标，并验证所有候选k/j的256-token和512-context长度限制；不通过则停止。无效样本仍只参与基础CoDi损失，覆盖率不随策略变化。

smoke使用256条固定样本（192有效、64无效），执行共同起点、相同配置双臂、扰动子代和不中断训练。验证相同配置双臂的模型/optimizer/scheduler/RNG逐位一致，断点恢复结果与不中断训练逐位一致，以及扰动确实改变参数更新。正式父子比赛继续审计起点checkpoint哈希、恢复步数、学习率与最初两个microbatch输入哈希。

## 最终评测与局限

搜索期间只读验证集。两条路径均完成epoch40且配置冻结后，分别评测GSM8K-Aug test、GSM-Hard、MultiArith600、SVAMP1000。沿用6 latent、batch128、greedy、答案上限256token、数值答案提取与MultiArith420/180分批边界。

这是单seed、有限邻域、短期训练收益导向的搜索。验证集重复使用4次，存在选择偏差；早期收益未必预测最终收益。不会据此宣称全局最优或latent/显式CoT的语义精确对齐。最终配置也尚未作为固定配置从头重训，须将动态配置轨迹的效果与固定setting本身的效果区分。

## 启动与产物

在worker上运行：

```bash
cd /home/tiger/codi-ab-c71db6e-20260909/repository
bash scripts/local_search/launch.sh
```

远端实验目录默认 `/home/tiger/codi-ab-c71db6e-20260909/random-disturb-20261015`。可用 `CODI_SEARCH_ROOT` 指定新的空目录；不要指向历史实验目录。已运行目录会从保存状态恢复，不会重置学习率计划或覆盖历史实验。

主要文件：`data-audit.json`、`smoke-verification.json`、`protocol.json`、`round-*-proposal.json`、`round-*-selection.json`、`frozen-policy.json`、`runs/*/progress.json`、`results.json`、`REPORT.md`。最终报告由控制器自动生成；`results.json`和`REPORT.md`出现之前，不表示实验已完成。
