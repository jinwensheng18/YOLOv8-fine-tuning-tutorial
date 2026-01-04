from ultralytics import YOLO
import pandas as pd
import numpy as np

# ===================== 核心配置（必须和你训练时一致） =====================
VAL_DATA_YAML = "./data.yaml"  # 你的81类数据配置
IMGSZ = 640  # 训练时的输入尺寸
BATCH = 32   # 训练时的batch_size
CONF = 0.001 # 评估置信度阈值（极低值保证召回）
IOU = 0.6    # NMS的IOU阈值
DEVICE = "5" # 训练用GPU

# ===================== 待评估的两个权重 =====================
OFFICIAL_WEIGHT = "./weights/yolov8n.pt"  # 官网80类权重路径
CUSTOM_WEIGHT = "path/to/train/best.pt"  # 你的81类权重路径

def get_full_class_names():
    """从data.yaml读取完整81类类别名（含wallet）"""
    import yaml
    with open(VAL_DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # 返回{0: 'person', ..., 79: 'toothbrush', 80: 'wallet'}
    return {i: name for i, name in enumerate(data["names"])}

def evaluate_weight(weight_path, weight_name, full_class_names):
    """
    评估单个权重，兼容80类（官网）和81类（自定义）
    :param weight_path: 权重路径
    :param weight_name: 权重名称（用于标注）
    :param full_class_names: 完整81类的{id: name}字典
    :return: 包含全部81类指标的DataFrame
    """
    print(f"\n========== 开始评估 {weight_name} ==========")
    # 加载模型
    model = YOLO(weight_path)
    
    # 执行验证（关键：禁用自动打印每类指标，避免KeyError）
    results = model.val(
        data=VAL_DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save_json=False,
        verbose=False,  # 关闭默认打印（避免遍历81类时报错）
        plots=False
    )
    
    # 提取模型实际包含的类别数（官网80类，自定义81类）
    model_class_num = len(results.names)  # 官网=80，自定义=81
    class_metrics = []
    
    # 遍历完整81类，逐个填充指标
    for cls_id in sorted(full_class_names.keys()):
        cls_name = full_class_names[cls_id]
        # 若类别ID超出模型范围（如官网权重的80类），指标设为0
        if cls_id >= model_class_num:
            precision = 0.0
            recall = 0.0
            map50 = 0.0
            map50_95 = 0.0
        else:
            # 提取该类的P/R/mAP50/mAP50-95（处理numpy类型，避免KeyError）
            precision = float(results.box.p[:, cls_id]) if cls_id < len(results.box.p) else 0.0
            recall = float(results.box.r[:, cls_id]) if cls_id < len(results.box.r) else 0.0
            map50 = float(results.box.ap50[:, cls_id]) if cls_id < len(results.box.ap50) else 0.0
            map50_95 = float(results.box.ap[:, cls_id]) if cls_id < len(results.box.ap) else 0.0
        
        class_metrics.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
            "weight_type": weight_name
        })
    
    # 转为DataFrame并返回
    df = pd.DataFrame(class_metrics)
    # 打印关键信息，确认评估成功
    print(f"{weight_name} 评估完成，有效类别数：{model_class_num}，总类别数：{len(full_class_names)}")
    return df

# ===================== 主流程 =====================
# 1. 获取完整81类类别名
full_class_names = get_full_class_names()

# 2. 评估官网权重（80类）
df_official = evaluate_weight(OFFICIAL_WEIGHT, "official_pretrain", full_class_names)

# 3. 评估自定义权重（81类）
df_custom = evaluate_weight(CUSTOM_WEIGHT, "custom_incremental", full_class_names)

# 4. 合并对比 & 计算变化率
df_merge = pd.merge(
    df_official,
    df_custom,
    on=["class_id", "class_name"],
    suffixes=("_official", "_custom")
)

# 计算变化率（避免除以0，官网指标为0时标记为"新增"）
def calc_change_rate(row):
    if row["mAP50_official"] == 0:
        return "新增"
    return round((row["mAP50_custom"] - row["mAP50_official"]) / row["mAP50_official"] * 100, 2)

df_merge["mAP50_change(%)"] = df_merge.apply(calc_change_rate, axis=1)
df_merge["recall_change(%)"] = df_merge.apply(
    lambda x: "新增" if x["recall_official"] == 0 else round((x["recall_custom"] - x["recall_official"]) / x["recall_official"] * 100, 2),
    axis=1
)

# 5. 保存对比结果
save_path = "weight_metrics_comparison.csv"
df_merge.to_csv(save_path, index=False, encoding="utf-8")
print(f"\n========== 对比结果已保存到：{save_path} ==========")

# 6. 打印关键类别对比
key_classes = ["wallet", "person", "toilet", "laptop", "book", "hair drier"]
df_key = df_merge[df_merge["class_name"].isin(key_classes)]
print("\n========== 关键类别对比 ==========")
print(df_key[["class_name", "mAP50_official", "mAP50_custom", "mAP50_change(%)"]])