# 连续三位置窗口续算（2026-10-22）

训练GPU2，worker4423662。远端目录：`/home/tiger/codi-ab-c71db6e-20260909/window-suffix-20261022`。从与历史baseline相同的原始初始化开始，保持主数据、seed11、GPT2+LoRA、batch64×accumulation2、40epoch/119840次更新、学习率0.003及cosine调度不变。

## 目标定义

K=6，m为原预处理后保留的显式CoT步数（依然移除最后一步）。j是从1开始的目标起始步骤编号，不是跳过的步数。

`center=ceil(k*m/6)`，`width=min(3,m)`，`start=min(max(center-1,1),m-width+1)`；候选为`start..start+width-1`。边界整体平移，不重复位置。

m=6时六个窗口依次为[1,2,3]、[1,2,3]、[2,3,4]、[3,4,5]、[4,5,6]、[4,5,6]。m=1所有k都使用[1]；m=2所有k都使用[1,2]；m=3所有k都使用[1,2,3]。m=0只训练原主任务。

每个k枚举全部窗口候选，不随机抽样。每条目标为`EOT + 第j步至第m步 + 答案 + EOS`。不提供前j-1步真实显式前缀；teacher forcing，EOT/padding不计loss，梯度通过复制的KV cache回传到latent与主模型。

## 精确归一化

对一个microbatch中N条有效样本：

`L_aux = (0.3/6) * sum_k [ (1/N) * sum_i (1/|J_ki|) * sum_j mean_token_CE(i,k,j) ]`。

即每条候选后缀先按有效token平均，样本内候选等权，再对有效样本平均，最后六个k各乘0.05相加。无有效样本时辅助项为0。短链候选少，不会因此降低该样本的总权重；长后缀不会因token多而占更大权重。

与GPU0旧实验相比，本实验还包含j=1整链目标、m=1样本，且归一化从全token混合平均改为上述分层等权。因此与旧GPU0比较不是仅改变窗口的严格单变量对照。本次仅启动用户指定的这个setting，不额外训练新的对照。

## 实现与验证

`src/window_suffix.py`：窗口枚举、打包、按序列均值加权的CE、六个latent hook。chunk64，非重入activation checkpoint保留dropout RNG；明确设置`accepts_loss_kwargs=False`，保持Trainer梯度累积缩放。

`prepare.py`：核对历史bank哈希，重建第1步开始的完整链目标；j>=2后缀逐字复用此前已完整审计的bank。主字段哈希与baseline一致；全部m分布与单步实验一致；禁止截断并检查上下文长度。检查完保存独立训练缓存和256条smoke缓存。

测试覆盖边界、m1/m2/m6、候选不重复、每个样本/候选/k的精确权重、完整forward与独立枚举loss及梯度一致、空样本、cache隔离、checkpoint RNG、梯度累积。随后GPU smoke训练、保存、验证通过才开始正式训练。

每epoch保存模型/优化器/调度器/RNG完整checkpoint，保留最近一次；每4epoch验证，最终评估validation、GSM8K test、GSM-Hard、MultiArith、SVAMP。再次运行`bash scripts/window_suffix/launch.sh`会核对缓存并从已保存checkpoint恢复。

## 评估资源协调

GPU0旧全后缀训练、GPU1单步训练保持运行。GPU1后续评估原本使用GPU2；在evaluate.py/evaluate_ood.py导入torch前加入仅作用于aligned-step-20261021 checkpoint且原请求GPU2的路由，改为GPU3，并持有GPU0实验的共享eval.lock直至进程退出。该变更只影响评估用卡，不改变模型输入、生成或评分规则。

本实验eval.lock链接到同一个共享锁，因而GPU0、GPU1、GPU2三个实验的评估均在GPU3串行执行，GPU2只承担新训练。评估路由的匹配范围、环境设置和排他锁有测试覆盖。当前运行的GPU1训练无需重启即可由其后续评估子进程读取路由。

状态：`launch.log`、`status.json`、`data-audit.json`、`runs/full/progress.json`、`runs/full/throughput.json`，最终`results.json`。本地实验目录仅为启动快照。

## 启动审计结果

主样本383368条，辅助有效320527条，无保留步骤62841条。每epoch辅助序列3430644条。最长目标175 tokens，保守padded上下文395 tokens，全部保留、无截断。17项测试通过，256样本GPU训练、保存、验证已完成，正式训练于2026-10-22启动。

平台worker指南列出最长96小时运行限制，当前worker创建于2026-10-20；若预计总训练时间超出剩余寿命，需在可用worker上从最近完整checkpoint续训，不能把后台进程启动视为保证本worker能跑完整个40epoch。缓存和checkpoint保存在共享持久目录。

前20次正式更新已核验：初始模型哈希和主输入哈希与baseline一致；六个k均参与，loss为有限值。实测2.647秒/update，预计40epoch纯训练88.10小时，峰值CUDA已分配26.02GiB。当前worker剩余寿命不足以覆盖该估计，需中途恢复续训；每epoch约2.2小时保存一次完整状态。GPU0/1/2均确认有训练负载，GPU3留给串行评估。本地`experiments/window-suffix-20261022`保存启动快照，不代表实时状态或最终结果。
