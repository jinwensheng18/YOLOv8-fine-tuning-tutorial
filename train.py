import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(model=r"./weights/yolo11n.pt", task="detect")  # 预训练权重路径
    model.train(
        data=r"./datasets/data.yaml",  # 配置文件路径
        imgsz=640,  # 输入图片大小
        # imgsz=800,
        epochs=150,  # 训练轮数
        batch=64,  # batch size
        workers=8,  # 多线程数
        device="2",  # 使用 GPU 设备编号
        optimizer="SGD",
        close_mosaic=3,  # 关闭马赛克增强的 epoch 数
        resume=False,  # 断点续训
        # freeze=16,
        mosaic=0.5,  # 马赛克增强概率
        project="runs/train",  # 结果保存路径
        name="exp_yolo11_wallet14",
        cache="disk",  # 开启缓存，预处理后的图片缓存到本地，加速后续 epoch 训练
        patience=20,  # 早停耐心值（50 个 epoch 无 val 精度提升则自动停止）
        save=True,  # 保存最佳模型（默认保存 val mAP 最高的模型）
        val=True,  # 每个 epoch 结束后验证（确保训练过程可监控）
        amp=False,
        # cls=1.2,
    )
