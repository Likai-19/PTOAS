# ConvPipeline 端到端验证样例

`conv_pipeline-pto.pto` 覆盖 conv tile 全链路，是"卷积流水线端到端可用"的
最短路径样例：

```
A 通道: GM fmap  --tload--> conv_tile(NC1HWC0) --set_fmatrix/rpt/padding--> timg2col --> LEFT
B 通道: GM weight--tload--> conv_tile(FRACTAL_Z) --textract--> RIGHT
计算:   tmatmul(LEFT, RIGHT) --> ACC --tstore--> GM
```

kernel 入参是三个 GM 指针（npu_validation 据此自动生成 device 侧
malloc/memcpy），view 在 kernel 内部构造，并在 view 类型上标注数据在 GM 中
已有的 conv 存储布局，使两条 tload 都是直通搬运、无需转换。

形状：fmap NC1HWC0 [1,1,3,4,16] f16（N=1,C1=1,H=3,W=4,C0=16），filter 2x2 /
stride 1 / pad 0 → 2x3 = 6 个输出位置，LEFT 16x32 valid 6x32；权重
FRACTAL_Z [2,1,16,16]（K=32,N=16）→ RIGHT 32x16；ACC 16x16 f32 valid 6x16
→ GM [6,16]。

## 运行真机验证

样例依赖 `--enable-insert-sync`：九条指令横跨 MTE2/MTE1/M/FIX 四条流水，
没有同步屏障时 tmatmul 会读到尚未搬完的 L0A/L0B，输出为乱码。`runop.sh`
默认即开启该选项。

```sh
ptoas --pto-arch=a5 --enable-insert-sync conv_pipeline-pto.pto -o conv_pipeline-pto.cpp
python <PTOAS>/test/npu_validation/scripts/generate_testcase.py \
    --input conv_pipeline-pto.cpp --testcase conv_pipeline \
    --output-root <out> --run-mode npu --soc-version <A5 SOC> \
    --aicore-arch dav-c310-cube
cd <out>/ConvPipeline/conv_pipeline && bash run.sh
```

`conv_pipeline_golden.py` 在 host 侧用 im2col + FRACTAL_Z→[K,N] 展开算参考
值；输入取小整数，f16 与 f32 累加都能精确表示，因此
`conv_pipeline_compare.py` 用 `atol=0.0` 做逐位比对。
