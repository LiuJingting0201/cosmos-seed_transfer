# 鱼眼相机多视角天气编辑 Pipeline

这是一套从开发机 `/root/liujingting/cosmos-gpt6-weather-20261030` 拉回的 DJI 鱼眼相机多视角实验编排代码，包含：

`survey → Astra author → Cosmos Reasoner → finalize → Astra edit → Astra audit → check → build-specs → Edge Transfer → verify → package`

入口示例：

```bash
python3 run_remote_case.py snow
```

## 目录

- `code/`：实验入口、Prompt/Astra bridge、Reasoner、spec 构建、Transfer、验证和打包脚本。
- `code/multiview/configs/snow.json`：DJI 鱼眼雪天实验配置。
- `examples/dji_snow_gpt6_v1/`：已完成的 DJI+Astra 雪天实验产物，包括四路 Prompt、author/reasoner/edit/audit provenance、Transfer spec、验证记录和四路生成视频。

## 已完成实验效果

`dji_snow_gpt6_v1` 使用四路 DJI 鱼眼道路视频，目标是将场景改成雪天：

- 四路 Transfer 均完成：4/4。
- 每路输入：80 帧、1088×1088、16 fps、5 秒。
- 每路输出：80 帧、960×960、16 fps、5 秒。
- 控制：Edge，weight=1.0，medium threshold。
- Prompt 编排：本地 `gpt-6-astra` 负责 author/edit/audit；Cosmos Reasoner 和 Cosmos Transfer 仍使用远端服务。

注意：实验报告明确标记 Edge control alignment 未做自动验证，`source_dimensions_preserved=false`；输出成功不等于鱼眼几何、天气语义和跨视角一致性已经通过自动评估，仍需要人工查看对比视频。

## 外部运行依赖

本目录是实验编排层，不包含 Cosmos 框架、模型权重或原始视频。运行时需要按配置访问：

- Cosmos checkout：`/root/sam31/cosmos`
- DJI 输入：`/root/sam31/data/dji_weather_rwy_standard/rwy_split2`
- Reasoner：`127.0.0.1:8001`
- Gateway/vLLM-Omni：`127.0.0.1:8002` / `127.0.0.1:8000`
- Astra 文件队列：由 `CODEX_BRIDGE_QUEUE` 指定

实验示例目录是已有结果快照，不包含模型权重和原始输入视频。
