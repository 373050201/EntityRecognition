# YOLO11s + RapidOCR工业流水线

该目录实现工业模块化实体检测基线：YOLO11s检测四类图元和Text框，MobileNetV3-Small根据图元裁剪二分类锁状态，RapidOCR识别Text裁剪区域，最后使用纯几何代价的匈牙利算法一对一组装完整实体。

## 使用顺序

1. 运行`prepare_dataset.py`，把当前实体JSON转换为五类YOLO标签；`train_yolo.py`启动时也会自动执行这一步。
2. 运行`train_yolo.py`，训练五类YOLO11s并生成`best_yolo11s.pt`。
3. 运行`train_lock_classifier.py`，训练锁状态小分类器并生成`lock_classifier_best.pt`。
4. 运行`tune_pipeline.py`，仅在验证集搜索图元、文字、OCR、锁分类和匹配距离参数。
5. 运行`evaluate_pipeline.py`，使用冻结模型和参数在测试集评估一次完整工业流水线。

YOLO11s类别依次为`RD、RS、GD、GS、Text`，锁状态不进入YOLO类别。锁状态分类器使用ImageNet预训练的MobileNetV3-Small，替换二分类头后约1.52M参数，输入为扩边后的96×96图元裁剪，输出未锁和已锁概率；训练集使用加权交叉熵缓解类别不平衡，并按验证集锁状态F1保存最佳模型。RapidOCR使用现成权重，不参与训练。

## 文件说明

- `config_yolo.yaml`：训练、推理、OCR、关联、验证调参和测试的统一配置。
- `requirements.txt`：YOLO11s、RapidOCR及评估所需的Python依赖。
- `data_yolo.yaml`：Ultralytics读取当前训练集、验证集和测试集的五类数据描述。
- `prepare_dataset.py`：从实体JSON生成四类图元和Text的YOLO标签。
- `dataset.py`：为验证和测试读取PIL原图及完整实体标注。
- `lock_classifier.py`：定义MobileNetV3-Small二分类器、图元扩边裁剪和批量锁状态推理。
- `lock_dataset.py`：从完整实体标注构造未锁/已锁图元裁剪数据集。
- `train_yolo.py`：微调五类YOLO11s，并按Ultralytics验证指标保存`best_yolo11s.pt`。
- `train_lock_classifier.py`：独立训练锁状态分类器，以验证集锁状态F1保存最佳模型。
- `pipeline.py`：执行YOLO检测、锁状态分类、RapidOCR识别和匈牙利实体关联。
- `tune_pipeline.py`：缓存验证集三模块输出，以实体F1搜索五个流水线参数。
- `utils.py`：坐标转换、严格实体指标和拆解指标计算。
- `evaluate_pipeline.py`：在测试集输出实体指标、拆解指标及端到端效率。
