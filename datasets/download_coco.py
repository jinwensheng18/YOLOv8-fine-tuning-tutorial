from pathlib import Path
import yaml  # 需导入 yaml 库解析配置文件

from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

# -------------------------- 关键配置（根据你的需求修改）--------------------------
YAML_PATH = "/home/at/jws/ultralytics/ultralytics/cfg/datasets/coco.yaml"  # 你的 yaml 配置文件路径（相对/绝对路径均可）
segments = False  # True=下载分割标注，False=下载边界框标注
DOWNLOAD_TEST = False  # 是否下载测试集（True=下载，False=不下载，节省空间）
# --------------------------------------------------------------------------------

# 读取 yaml 配置文件，获取数据集根目录
with open(YAML_PATH, "r", encoding="utf-8") as f:
    yaml_config = yaml.safe_load(f)  # 用 safe_load 避免安全风险
dir = Path(yaml_config["path"])  # 数据集根目录（从 yaml 中读取）
dir.parent.mkdir(parents=True, exist_ok=True)  # 确保父文件夹存在（避免下载时路径错误）

# 1. 下载 COCO 标注文件（分割/边界框）
label_url = ASSETS_URL + ("/coco2017labels-segments.zip" if segments else "/coco2017labels.zip")
download([label_url], dir=dir.parent)  # 标注文件下载到数据集根目录的父文件夹
print("✅ 标注文件下载完成！")

# 2. 下载 COCO 图片文件（训练集+验证集，可选测试集）
image_urls = [
    "http://images.cocodataset.org/zips/train2017.zip",  # 训练集（19G，必下）
    "http://images.cocodataset.org/zips/val2017.zip",    # 验证集（1G，必下）
]
if DOWNLOAD_TEST:
    image_urls.append("http://images.cocodataset.org/zips/test2017.zip")  # 测试集（可选）

# 图片下载到「数据集根目录/images」文件夹，3线程加速
image_dir = dir / "images"
image_dir.mkdir(parents=True, exist_ok=True)  # 确保图片文件夹存在
download(image_urls, dir=image_dir, threads=3)
print("✅ 所有图片文件下载完成！")

# 3. 提示：下载后会自动解压（ultralytics 的 download 函数默认自动解压）
print(f"\n📁 数据集最终路径：{dir}")
print(f"  - 标注文件：{dir.parent / 'coco2017labels-segments' if segments else dir.parent / 'coco2017labels'}")
print(f"  - 图片文件：{image_dir}")