# 按seed分组的规范数据集

当前根目录G:/FutsalMOT_Dataset/v1_no_numbers_4k。dataset.json的schema=futsalmot_seed_dataset、version=1、layout=seed_flat_v1。

```text
dataset.json
splits.json
README.md
shared/{generation_recipe.json,quality_policy.json,layout_migration.json,calibrations/,catalogs/,runtime_profiles/}
groups/seed_XXXXXXXX/
  group.json
  clip.json
  trajectory.jsonl
  checksums.json
  cameras/CineCam_01/
    view.json
    annotations.jsonl
    rgb/000001.png
    mask/000001.png
```

旧clips/clip_000用于同seed多个片段；本轮每seed只有一个连续窗，移除两层目录。保留逻辑clip ID和clip.json，不丢失时间范围或准入证据。平铺布局拒绝追加第二个不同片段。

## 引用与信息去重

dataset.groups[].path相对数据集根；group.clips[].path相对seed目录，当前为clip.json。view.calibration_ref和外观catalog_ref相对数据集根。按引用读取，不能硬编码父目录深度；解析后必须仍在所属根目录内。

group保存一次实体属性、外观、天气、随机种子及split；clip保存帧数、任务ID、源窗口与准入；trajectory只保存一次世界轨迹。frame_index从1开始对应同名PNG，trajectory重基step从0开始。

每视角objects仅包含entity_id、bbox_xyxy、in_frame、truncated、visibility、visible_pixel_count、bbox_source及已有segmentation。bbox是可见实例像素半开框，完全不可见为null，不等同于amodal人体框。team、role、外观和世界坐标不逐相机重复。

checksums.json的路径相对clip.json所在目录，覆盖轨迹、clip、view/annotations及全部RGB/mask，不包含自身和group.json。

## 迁移和验证

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m grf_ue_bridge.dataset_layout G:/FutsalMOT_Dataset/v1_no_numbers_4k --backup D:/backup/layout
# dry-run后明确添加 --apply
```

迁移要求每组恰好一个片段，先检查目标冲突并备份文本，再同卷移动目录，修正group、full审计anchor和clip文本SHA；图像身份、编码和字节不变。历史来源记录不伪装成新布局生成。

verify_clip默认full重读全部SHA；metadata核验文本SHA及图像头；structure只核验文本、存在性、帧覆盖和准入引用，复用已发布证据，不重读媒体。structure不能发现媒体内部损坏，也不是像素审计。

旧seed_clips_v1继续兼容读取；向平铺版本发布新seed会沿用平铺布局。

## 研究派生

研究数据仅放D:/projects/FutsalMOT_Research。YOLO与MOT都从同一正式标注派生，通过--reference-images目录联接直接引用正式RGB，不跨盘复制整套4K。清单必须保留images的词法路径，不能resolve到G盘rgb，否则YOLO无法匹配派生labels。

同seed全部机位同split，本轮均train。真实验证和测试按比赛隔离，不随机拆同轨迹邻帧。C从32个完整视角按相机和时间均衡选300张；完整MOT连续序列不被训练采样截短。
