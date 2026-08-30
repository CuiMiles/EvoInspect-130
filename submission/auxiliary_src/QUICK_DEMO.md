# Quick Demo

`run_demo.sh`接受PNG/JPEG，使用OpenCV解码并重采样到256×256，再通过ONNX Runtime运行
EfficientAD-M FP16异常图。脚本默认优先CUDA，若不可用则回退CPU；每次输出实际provider。

```bash
./run_demo.sh path/to/input.png [output_directory]
```

结果字段：`score`、`threshold`、`decision`、`input_resolution`、`model_input_resolution`、
`provider`、`latency_ms`和`model_sha256`。该入口是代表类别功能演示，不是15类通用模型。

