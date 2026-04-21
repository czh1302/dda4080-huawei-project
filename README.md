# dda4080 — Long-Tail Driving Scene Discovery

基于 [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)(Qwen2.5-VL 架构)在 NuScenes 关键帧上做 long-tail(罕见场景)的表示提取与分析。

仓库内含两个独立脚本:

| 脚本 | 作者 | 作用 |
|---|---|---|
| `scripts/extract.py` | yike | 对每个 clip 的 6 路摄像头各跑一次 VLM,抽取最后一层 MLP 的输入/输出激活,并让模型生成 `describe the video in detail` 的文字输出。 |
| `scripts/pipeline.py` | zihua | 端到端流程:抽取 `last MLP` 前的隐状态 → SAE(Sparse Autoencoder)训练 → 罕见场景(tail)分类/可解释性分析。 |

两个脚本相互独立,可分别运行。

---

## 1. 环境

- Python ≥ 3.10
- CUDA GPU(bf16 推理,显存约需 ≥ 24 GB;`pipeline.py` 用 `cuda:0`)
- 磁盘:模型 ~16 GB + NuScenes keyframes 解压后 ~40 GB + 输出若干 GB

安装依赖:

```bash
pip install -r requirements.txt
```

> `transformers` 版本需支持 `Qwen2_5_VLForConditionalGeneration`(≥ 4.49)。

## 2. 需要准备的外部数据(**不在仓库里**)

项目根目录下需要这样的布局:

```
dda4080_huawei_project/
├── models/
│   └── Cosmos-Reason1-7B/                 # 从 HuggingFace 下载
├── dataset/
│   └── nuscenes/
│       ├── v1.0-trainval01_keyframes.tgz  # NuScenes 官方关键帧分片(01~10)
│       ├── v1.0-trainval02_keyframes.tgz
│       └── ...
└── nuscenes_annotation_result/
    └── vlm_annotated_clips.json           # VLM 标注出来的 clip 列表(含 tail_score)
```

获取方式:

1. **Cosmos-Reason1-7B**
   ```bash
   huggingface-cli download nvidia/Cosmos-Reason1-7B \
       --local-dir models/Cosmos-Reason1-7B
   ```

2. **NuScenes keyframes tgz**:从 [nuScenes 官网](https://www.nuscenes.org/nuscenes) 下载 `v1.0-trainval0{1..10}_keyframes.tgz`,放到 `dataset/nuscenes/` 下,**不需要手动解压**(脚本会自动 untar)。

3. **`vlm_annotated_clips.json`**:由上游 VLM 标注平台产出,结构为 `list[dict]`,每个 clip 含:
   - `global_clip_index`、`scene_name`、`tail_score`(0~5,越大越罕见)
   - `frames[*].channels.{CAM_FRONT, CAM_FRONT_LEFT, ...}` → 指向 `samples/CAM_*/xxx.jpg` 的绝对路径(历史路径前缀为 `/data2/visitor/czh/nuscenes_keyframes/`,`pipeline.py` 会自动重映射到本地)

## 3. 运行

### 3.1 `extract.py` — 抽取 MLP 表示 + 生成文本

默认对 NuScenes parts 04/05/06 运行:

```bash
cd scripts
python extract.py \
    --parts 4 5 6 \
    --tgz-dir ../dataset/nuscenes \
    --extract-root ../dataset/nuscenes/nuscenes_extracted \
    --annotations ../nuscenes_annotation_result/vlm_annotated_clips.json \
    --model-dir ../models/Cosmos-Reason1-7B \
    --output-dir ../output/cosmos_extract_output \
    --frames-per-camera 16 \
    --resume          # 中断后可续跑
```

产物(在 `--output-dir` 下):

| 文件 | shape / 格式 | 含义 |
|---|---|---|
| `last_mlp_output.npy` | `[N, 6, hidden_size]` float32 | 最后一层 MLP 输出,按 token 做 mean-pool |
| `last_mlp_output_last_token.npy` | `[N, 6, hidden_size]` | 同上,取最后一个 token |
| `last_mlp_intermediate.npy` | `[N, 6, mid_dim]` | 最后一层 MLP 的中间层(`down_proj` 的输入),mean-pool |
| `last_mlp_intermediate_last_token.npy` | `[N, 6, mid_dim]` | 同上,取最后一个 token |
| `final_text_output.jsonl` | 每行一个 clip | 每个相机的文字描述 |
| `meta.json` | list | 每个 row 的 `global_clip_index`、`scene_name`、`tail_score` 等 |

其中 `N` 是保留下来的 clip 数(6 个相机全部取不到图的 clip 会被跳过),`6` 对应 6 路相机。

### 3.2 `pipeline.py` — 端到端 SAE 流程

默认对 parts 01/02/03 运行:

```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 python pipeline.py --parts 1 2 3
```

关键开关:

- `--skip-extract`:已经有 `pipeline_output/features.json` 时跳过第 2 步直接训 SAE
- `--frames-per-camera`、`--max-pixels`:控制 VLM 输入的帧数与分辨率
- `--hidden-dim`、`--encoder-layers`、`--tail-ratio`、`--margin` 等:SAE / 训练超参
- 详见 `parse_args()`

流程(7 步,日志里有分段标题):

1. 用 tgz 文件列表建图片白名单
2. 加载 Cosmos,抽取 `last MLP` **前**的 hidden state(`[seq_len, 3584]`),对视觉 token mean-pool 得到每路相机的 embedding
3. Pre-SAE 分析(raw L2 / mean-diff / logistic regression 的 AUC baseline)
4. 训练 `DeepLongTailSAE`(pairwise margin loss + mixup + tail oversampling)
5. Post-SAE 分析(`z_t` L2 的 AUC、二分类指标)
6. 语义可解释性:找 top-k 神经元对应的 top clips
7. 汇总 Pre vs Post 指标

产物(写到 `scripts/pipeline_output/`):

- `features.json` — 每个 clip 的 6 路相机 + 聚合 embedding(**约 170 MB**)
- `best_model.pth` — 最优 SAE 权重
- `metrics.json` — Pre/Post 指标
- `interpretation.json` — top 神经元的场景解释

## 4. 仓库里**没有**放的东西

为避免仓库过大,以下内容被 `.gitignore` 排除,需要按第 2 节自行准备或重跑:

- `models/`、`dataset/`:大模型与原始数据
- `output/`:2.1 GB 的历史产物
- `scripts/pipeline_output/`:`pipeline.py` 默认输出目录
- `nuscenes_annotation_result/vlm_annotated_clips.json`:66 MB 的上游标注文件
- `*.log`、`__pycache__/`、`.cache/`、conda 环境等

如果需要参考历史产物的文件结构,可以看之前机器上的 `output/yike_script_output/` 和 `output/zihua_script_output/` 目录。

## 5. 常见坑

- **`vlm_annotated_clips.json` 里的路径写死了 `/data2/visitor/czh/...`**。`pipeline.py` 会把它重映射到 `dataset/nuscenes/samples/...`;`extract.py` 的做法是只取 `samples/` 后面的相对路径再拼 `--extract-root`。换机器/换路径时两个脚本都能跑,但注意保留 `samples/...` 这段子路径结构。
- **bf16 + `device_map="auto"`(extract.py)/ `"cuda:0"`(pipeline.py)**:多卡机器上 `extract.py` 会自动切分,`pipeline.py` 固定单卡。
- **Resume**:`extract.py` 加 `--resume` 从 `meta.json` 的最后一条 `global_clip_index` 续跑;`pipeline.py` 通过 `pipeline_output/extract_checkpoint.json` 自动续跑特征抽取阶段。
- **显存尖峰**:`extract.py` 在抽完激活后会显式 `captures.clear()` 再 `generate`,避免文字生成时 OOM。如果还是 OOM,调低 `--frames-per-camera` 或 `--max-pixels`。
