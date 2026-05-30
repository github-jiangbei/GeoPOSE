# GeoPOSE Project Architecture

本文档用 Mermaid 描述当前项目的代码结构、训练/推理数据流，以及加入 geometry prompt 后的 FinePOSE 模型内部流程。

![GeoPOSE geometry-prompt architecture](ARCHITECTURE_FIGURE.svg)

## 1. 模块总览

```mermaid
flowchart TB
    subgraph Entrypoints["入口脚本"]
        Main["main.py<br/>Human3.6M 训练/评估/可视化"]
        MainGT["main_gt.py<br/>GT/旧流程入口"]
        Main3DHP["main_3dhp.py<br/>MPI-INF-3DHP 入口"]
    end

    subgraph Data["数据与批处理"]
        H36M["common/h36m_dataset.py<br/>Human36mDataset"]
        Mocap["common/mocap_dataset.py<br/>MocapDataset 基类"]
        Skeleton["common/skeleton.py<br/>骨架拓扑"]
        Gen["common/generators.py<br/>Chunked/Unchunked Generator"]
        Gen3DHP["common/generators_3dhp.py<br/>3DHP Generator"]
    end

    subgraph Geometry["相机与几何"]
        Camera["common/camera.py<br/>坐标变换/投影"]
        Quaternion["common/quaternion.py<br/>旋转工具"]
        GeoPrompt["common/geometry_prompt.py<br/>TCN 粗姿态 + Geometry Prompt"]
    end

    subgraph Model["模型"]
        FinePOSE["common/finepose.py<br/>扩散采样与训练封装"]
        MixSTE["common/mixste_finepose.py<br/>Denoiser / MixSTE2"]
        FinePOSE3DHP["common/finepose_3dhp.py"]
        MixSTE3DHP["common/mixste_finepose_3dhp.py"]
    end

    subgraph Support["损失/日志/工具/可视化"]
        Loss["common/loss.py<br/>MPJPE / P-MPJPE / Reprojection select"]
        Args["common/arguments.py<br/>命令行参数"]
        Utils["common/utils.py"]
        Log["common/logging.py"]
        Vis["common/visualization.py"]
        Graph["common/graph_utils.py"]
    end

    Main --> Args
    Main --> H36M
    H36M --> Mocap
    H36M --> Skeleton
    H36M --> Camera
    Camera --> Quaternion
    Main --> Gen
    Main --> FinePOSE
    FinePOSE --> GeoPrompt
    FinePOSE --> MixSTE
    GeoPrompt --> Camera
    Main --> Loss
    Main --> Log
    Main --> Vis
    Main3DHP --> Gen3DHP
    Main3DHP --> FinePOSE3DHP
    FinePOSE3DHP --> MixSTE3DHP
```

## 2. 训练数据流

```mermaid
flowchart LR
    Args["parse_args()"] --> Load3D["加载 data_3d_<dataset>.npz"]
    Load2D["加载 data_2d_<dataset>_<keypoints>.npz"] --> Normalize2D["normalize_screen_coordinates"]
    Load3D --> CameraSpace["world_to_camera<br/>得到 camera-space 3D"]
    CameraSpace --> RootSplit["保存 root trajectory<br/>训练目标 root-relative"]
    Normalize2D --> Fetch["fetch(subjects_train/test)"]
    RootSplit --> Fetch
    Fetch --> Generator["ChunkedGenerator_Seq<br/>按 number_of_frames 切片"]
    Generator --> Batch["batch_2d, batch_3d, camera_intrinsic, action"]
    Batch --> Text["CLIP tokenize action/pre_text"]
    Batch --> FinePOSETrain["FinePOSE.forward(train)"]
    Text --> FinePOSETrain
    FinePOSETrain --> Pred3D["denoised 3D pose"]
    FinePOSETrain --> Coarse3D["TCN coarse 3D pose"]
    Pred3D --> Loss3D["MPJPE(final, GT)"]
    Coarse3D --> CoarseLoss["MPJPE(coarse, GT)"]
    Loss3D --> Total["total loss"]
    CoarseLoss --> Total
    Total --> Optim["AdamW 更新"]
```

## 3. FinePOSE + Geometry Prompt 内部结构

```mermaid
flowchart TB
    Input2D["输入 2D pose<br/>B x F x J x 2/3"]
    Input3D["训练: GT 3D<br/>推理: random noise"]
    CameraParam["相机内参<br/>f, c, distortion"]
    RootTraj["root trajectory / root depth"]
    TextCond["action text + prompt text"]

    subgraph Coarse["Step 1: TCN 粗 3D"]
        TCN["TemporalConvPose<br/>1D Temporal Conv Residual Blocks"]
        CoarsePose["coarse root-relative 3D"]
    end

    subgraph Prompt["Step 2-4: 几何 Prompt"]
        AbsPose["coarse 3D + root trajectory<br/>得到 camera-space absolute pose"]
        Project["project_to_2d"]
        Error["projection error<br/>projected 2D - input 2D"]
        Ray["camera ray"]
        DepthConf["depth + confidence"]
        GeoToken["15D geometry prompt / joint"]
    end

    subgraph Diffusion["Step 5-6: 条件扩散去噪"]
        QSample["训练: q_sample 加噪"]
        DDIM["推理: DDIM sampling loop"]
        Denoiser["MixSTE2 denoiser"]
        Spatial["Spatial joint tokens<br/>2D + noisy 3D + geometry prompt"]
        Temporal["Temporal blocks + text cross attention"]
        Output["修正后的 3D pose"]
    end

    Input2D --> TCN
    TCN --> CoarsePose
    CoarsePose --> AbsPose
    RootTraj --> AbsPose
    AbsPose --> Project
    CameraParam --> Project
    Project --> Error
    Input2D --> Error
    Input2D --> Ray
    CameraParam --> Ray
    AbsPose --> DepthConf
    Error --> GeoToken
    Ray --> GeoToken
    DepthConf --> GeoToken
    CoarsePose --> GeoToken
    Project --> GeoToken
    Input2D --> GeoToken

    Input3D --> QSample
    QSample --> Denoiser
    DDIM --> Denoiser
    Input2D --> Denoiser
    GeoToken --> Denoiser
    TextCond --> Denoiser
    Denoiser --> Spatial
    Spatial --> Temporal
    Temporal --> Output
    Output --> DDIM
```

## 4. 推理与评估流程

```mermaid
sequenceDiagram
    participant Main as main.py:evaluate()
    participant Gen as UnchunkedGenerator_Seq
    participant Model as FinePOSE(eval)
    participant Geo as GeometryPromptBuilder
    participant Den as MixSTE2
    participant Loss as loss.py

    Main->>Gen: next_epoch()
    Gen-->>Main: cam, batch_3d, batch_2d, action
    Main->>Main: eval_data_prepare()
    Main->>Model: inputs_2d, inputs_3d, camera, root_trajectory, text
    Model->>Geo: TCN coarse pose + projection geometry
    Geo-->>Model: geometry prompt
    loop sampling_timesteps
        Model->>Den: noisy 3D + 2D + text + geometry prompt
        Den-->>Model: predicted x_start
        Model->>Model: DDIM update
    end
    Model-->>Main: proposals per sampling step
    Main->>Loss: MPJPE / reprojection-based selection
    Loss-->>Main: protocol metrics
```

## 5. 关键张量约定

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| `inputs_2d` | `B x F x J x 2/3` | 归一化 2D 关节，第三通道可作为置信度 |
| `inputs_3d` | `B x F x J x 3` | root-relative 3D 训练目标 |
| `inputs_traj` | `B x F x 1 x 3` | root 轨迹，用于恢复 camera-space absolute pose |
| `camera_params` | `B x 9` | Human3.6M 内参向量 |
| `coarse_pose` | `B x F x J x 3` | TCN 输出的粗 3D pose |
| `geometry_prompt` | `B x F x J x 15` | 投影误差、相机射线、深度、置信度等几何条件 |
| `predicted_3d_pos` | 训练: `B x F x J x 3`；推理: `B x T x H x F x J x 3` | 扩散 denoiser 输出 |
