# SGFormer及比较模型适配记录

## 代码与许可

作者：[qitianwu/SGFormer](https://github.com/qitianwu/SGFormer)，NeurIPS2023主会。固定复用提交：`3578e101c701491ce068bf26a9b029d2134903be`，来源`large/ours.py`。原MIT许可证全文保留在`models.py`顶部，Copyright (c) 2025 Qitian Wu。方法代码不是通过执行外部下载脚本取得。

保留作者的线性投影、带bias的Q/K/V、整体张量范数、LayerNorm、BatchNorm、ReLU、dropout和残差。全局1层1头，默认局部1层；`layers`控制局部分支层数，局部2层时仍复用作者“每层加入初始投影”行为，不改成新的层间残差。原分类线性head作为64维原始嵌入投影，没有分类softmax。共同点积除以sqrt(64)的关系解码和BCE由runner实现。

## 明确的实现修改

1. 删除`torch_sparse`扩展依赖，将`SparseTensor(row=col,col=row) @ X`替换为等价带权`index_add`。incoming degree按作者`degree(col,N)`计算，权重为1/sqrt(d_source*d_destination)，无穷值置0。runner提供双向关系边，此函数不加自环。重复边求和。
2. 全局Q/K计算转float32，除以整体范数的`clamp_min(1e-8)`，避免零范数NaN。不改成PyG的逐节点归一化。全局attention不显式产生N×N矩阵。
3. 统一属性frontend：代谢物2048维、疾病1536维分别可训练线性投影到128维，没有额外激活或规范化。加2×128类型embedding，维数仍128；这是两个类型共享的标记，不是节点ID记忆表。
4. 可学习门控输入3维：log1p(当前目标关联度)＋两维节点类型one-hot，网络3→8→1（中间ReLU，最终sigmoid），共41参数。度为0时显式乘0，使混合结果保留完整全局分支。degree来自runner当前上下文，不计人工自环。
5. `global_only`不创建局部分支和门控；不是仅使用原作者`use_graph=False`而仍保留闲置图参数。重置函数也重置最后输出fc，修补原作者遗漏。每seed仍应从头创建模型。
6. `full`/`gate_uniform`采用同一个可学习门控结构；`sgformer`/`fixed_budget`/`fixed_uniform`采用同一个固定局部alpha；`degree_budget`使用k/(k+c)，不建门控MLP。遮蔽方式只由runner控制，模型不偷偷改变输入图。

## 少量主比较

各方法共用相同原始属性和frontend、节点顺序、64维输出、关系解码、标签/候选集合、划分和停机预算，保留各模型传播机制。参数不同应如实报告，不人为称完全相同容量。

- **MLP**：128→hidden→64，两层线性，中间ReLU/dropout；不读取关联图。
- **SGC**：PyG官方`SGConv(128,64,K=layers,cached=False)`，无额外隐层非线性；保留官方规范化及自身自环。由于训练frontend和query/预算改变上下文，不能沿用一次预计算成本或旧缓存。
- **GATv2**：作者推荐的PyG官方`GATv2Conv`，单头、`concat=False`，`layers`控制1或2层；中间层ReLU/dropout，末层输出原始64维。保留官方自身自环和attention dropout，无分类softmax。
- **固定SGFormer**：同作者骨干、相同输出投影、两分支按alpha混合；任何额外遮蔽由runner按实验名称区分。

PyG为MIT，SGC来源ICML2019、GATv2来源ICLR2022，不能将新任务适配称为原作者已报告CTD结果。SGC/GATv2内部自环是模型算子机制，availability degree只依据外部目标关系边，不能把自环计为证据。

## 参数核对

2048→128：262,272；1536→128：196,736；类型embedding：256；frontend共459,264。

128输入、隐藏/输出64、1层全局/局部时，作者编码核心37,824；真正仅全局核心25,152。完整门控方法参数为459,264＋37,824＋41＝**497,129**。固定SGFormer为497,088；仅全局为484,416。这些值将在小图测试中通过实际参数统计确认；其他隐藏/层数须实际重新计算。

## 工程检查与边界

实际特征流程先以sanitize=False解析源SMILES，再SanitizeMol及AssignStereochemistry；它保留源显式H原子，未另行RemoveHs，因此不能称为默认重原子Morgan。独立复核发现234个最终节点中69个与默认MolFromSmiles去除显式H后的Morgan不同；两种表示的完整InChIKey一致，按已冻结流程独立重算的指纹全部匹配。所有方法共享本轮特征，该表示选择没有按正式测试分数调整。若后续统一去H，须作为新的预设数据/特征版本，不能混用本轮成绩。

`test_models.py`使用模拟小图，只核对接口和工程正确性，不是科研效果：所有10个方法的有限前向/损失/反向梯度；无闲置可训练参数；仅两个类型embedding；参数计数；零度门控及完整全局保留；真正删除局部模块；index_add和作者规范化稠密参照相等（含重复边/孤立节点）；零范数/空图稳定；固定及度数规则门控；SGC不缓存；head重置；非默认隐藏/层数。

可在root准备的独立实验环境运行：`python experiments/ctd_sgformer/test_models.py`。模型文件本身不安装包，不联网，不训练完整数据；真实指标须由runner加载冻结数据、固定划分后计算。
