# 垃圾目标检测训练与服务

这是基于 TACO 和 YOLO11n 的训练、验证、权重保存、API 推理和浏览器展示系统。当前小样本模型效果较弱，适合作为实验与工程作品，不能作为成熟垃圾分类设备宣传。

```powershell
python scripts/download_models.py
python scripts/prepare_taco.py --count 32 --seed 42
python experiments/train_detection.py --epochs 40 --name taco-40
$env:AI_YOLO_MODEL="runs/taco-40/weights/best.pt"
python -m uvicorn ai_xr.api:app --host 127.0.0.1 --port 8000
```

在浏览器 YOLO 页上传图片，得到检测框、类别、置信度和本次调用时延；检测上下文可继续送入 XR 助手。默认 `models/yolo11n.pt` 是 COCO 预训练权重，不是垃圾分类模型，必须显式切换才使用 TACO 训练权重。

## 数据与指标

从公开 TACO 标注和 Flickr URL 下载实际照片，固定种子按图像划分 24 张训练与 8 张验证，映射为 bottle/can/cup/bag/other_litter 五类。`evidence/taco-split.json` 保存图像 ID、划分、来源与文件校验值。照片不上传 GitHub。

3 epoch 基线 mAP50=0。40 epoch 后，在仅 8 张、9 个目标的验证集上，mAP50=0.0671、mAP50-95=0.0464、Recall=0.0625。虽然宏平均 Precision 数字较高，但召回极低，不能挑选 Precision 作为“识别准确率”。同一验证集用于选择 best 权重，并非独立最终测试集。

10 次单图 pipeline 延迟取中位数；其倒数只是单图吞吐近似，不是相机、编解码、网络、Unity 渲染全部在内的端到端视频 FPS。硬件为 i5-11400H，PyTorch CPU 运行；同期负载会影响绝对时延。

改进路径：增加足够且类别均衡的数据，按拍摄来源划分独立测试集，检查错误类别映射，加入小目标增强并做输入尺寸/训练轮次消融。记录假阴性，不通过反复挑选验证集抬高数字。

来源：[TACO](https://github.com/pedropro/TACO)、[Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)。项目使用框标注做检测，不声称复现 TACO 实例分割。
