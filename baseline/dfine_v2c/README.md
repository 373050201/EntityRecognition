# D-FINE-V2C

实验4：将 V2-C 的双区域复合实体查询迁移到官方 D-FINE-S。

## 目录职责

- `model.py`：组装 D-FINE-S、锁状态头和 V2-C text-only OCR 头。
- `loss.py`：保留 D-FINE 的 FDR/GO-LSD 检测损失，并增加文字框、锁状态和 CTC 损失。
- `train.py`：训练和验证，按验证集实体 F1 保存 `best.pt`。
- `evaluate.py`：在测试集计算实体指标、拆解指标、损失、FPS、FLOPs 和峰值显存。
- `config.yaml`：数据、模型和训练超参数。
- `requirements.txt`：D-FINE 相对当前项目新增的运行依赖。

官方 D-FINE 源码位于 `third_party/dfine`，预训练权重默认位于
`weights/dfine_s_coco.pth`。

图元框和文字框分别维护独立 FDR 分布。两个分支都包含 pre-box、逐层
分布残差细化、FGL 监督和由最终层指导浅层的 DDF 自蒸馏；二者共享实体
匹配结果，但不共享 reference、采样特征或边界分布状态。

## 运行

```powershell
python baseline/dfine_v2c/train.py
python baseline/dfine_v2c/evaluate.py
```

训练检查点写入当前目录的 `best.pt` 和 `last.pt`。评估脚本默认读取
`best.pt` 并使用 `synthetic_test`。
