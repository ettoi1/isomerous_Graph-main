# Brain Graph MoE MVP

这是一个用于基于 fMRI ROI 时间序列进行脑疾病分类的最小可运行版本（MVP）代码骨架。我们将受试者的 ROI 时间序列转化为图结构，并通过包含社区模块、超图编码器和混合专家（MoE）层的模型进行二分类。

## 项目结构

```
your_project/
├─ data/
│  ├─ raw/          # 原始 ROI 时间序列（仅占位）
│  ├─ processed/    # 预处理缓存（仅占位）
│  └─ splits/       # train.txt / val.txt / test.txt
├─ src/
│  ├─ config.py
│  ├─ dataset/
│  │  ├─ roi_dataset.py
│  │  └─ collate.py
│  ├─ preprocess/
│  │  ├─ compute_edge_features.py
│  │  ├─ build_graph.py
│  │  └─ build_hypergraph.py
│  ├─ models/
│  │  ├─ community_module.py
│  │  ├─ hypergraph_module.py
│  │  ├─ moe_layer.py
│  │  ├─ readout.py
│  │  └─ brain_gnn.py
│  ├─ training/
│  │  ├─ losses.py
│  │  ├─ metrics.py
│  │  └─ train_loop.py
│  ├─ utils/
│  │  ├─ seed.py
│  │  ├─ logging.py
│  │  └─ masking.py
│  └─ main_train.py
└─ README.md
```

## 快速开始

默认配置会自动生成一个可控的 synthetic 数据集（保存到 `your_project/data/raw_synth/`），并基于该数据集完成端到端冒烟测试，覆盖图构建、超图、MoE、社区分配等模块。

如需使用真实数据，可将 `config.use_synthetic` 设为 `False` 并准备真实拆分文件。

1. 运行训练脚本：

```bash
python -m src.main_train
```

> 💡 提示：如果当前环境缺少 PyTorch 或 NumPy，脚本会友好地给出依赖缺失提示并提前退出，不会抛出异常。安装对应依赖后即可触发真正的训练流程。

当前数据加载、偏相关、动态 FC 方差与互信息计算均为占位实现，待替换为真实算法。训练循环使用 batch size = 1；当需要更大批次时，请扩展 `collate_fn`。Synthetic 数据用于验证整个训练闭环是否正常工作。

## 未来工作 TODO

- [ ] 使用真实的偏相关与互信息估计方法替换占位实现。
- [ ] 实现真正的动态功能连接 (dFC) 方差计算。
- [ ] 引入超图注意力或更复杂的聚合机制。
- [ ] 增加社区正则项与容量限制，更好地约束 MoE。
- [ ] 集成更强的日志记录、早停与 TensorBoard 可视化。
- [ ] 将图构建迁移到 PyTorch Geometric，支持更高效的批处理。
