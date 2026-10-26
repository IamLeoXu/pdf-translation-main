# OCR 排版与本地 OCR 修复

本轮修改 OCR 路径；text_based=true 的文字层翻译代码及原 Dify 接口不变。实际使用的本机项目目录就是本文件所在目录。

## 已修改

- 保留 Paddle 返回的完整原始识别框，不再用估算的 12pt 行高缩短原文范围。修复原文末行露出、残影和覆盖不完整。
- 在绘制译文前先覆盖全部原文区域，覆盖范围使用完整源框，而不是缩小后的文字排版框。
- 按同列下一个段落限制当前段落高度。统一正文字号和“短文字允许溢出”不再跳过实际 Typst 测量。
- 短段落可以使用下一段之前的安全空隙；必要时按真实 Markdown 高度缩小字号，保留字形之间的间隔。
- local 默认使用镜像内的 CPU PaddleOCR PP-StructureV3，无需另外启动 8080 服务或提供 OCR token。云端 OCR 路径保持原接口。
- OCR 使用独立 Python 环境，避免改变已经正常的 text 翻译依赖。固定 PaddlePaddle 3.2.2、PaddleOCR 3.5.0、PaddleX 3.5.0；不加载未启用的图表大模型。
- 本地 OCR 子进程失败时会回显具体原因，并遮盖已知环境凭据。

## 如何使用

在当前项目目录执行：

```powershell
docker compose up -d --build unified
```

首次构建会下载本地 OCR 依赖和模型，比原来的云端专用镜像更大、构建更久。构建阶段会实际执行 CPU 推理检查。预热的源语言为英语、德语、意大利语、简体/繁体中文、日语、土耳其语和阿拉伯语；其他 Paddle 支持的源语言首次使用可能需要下载对应识别模型。目标翻译语言仍按每次请求的 lang_out 动态选择。

测试本地 OCR 时：

- text_based=false
- ocr_provider=local
- lang_in 填真实原文语言，例如 de
- lang_out 填需要的目标语言，例如 it
- 模型的 key、base_url、model 仍使用原来能工作的配置；本地 OCR 仅替代识别步骤，不替代翻译模型。

云端 OCR 仍选择 cloud。已生成的 PDF 不会自动改变，需要提交新任务。

原始 Dify YAML 的路径和字段保持兼容，不需要为本次修复修改。当前实际 .env 没有改动；旧的 RETAIN_LOCAL_OCR_URL 占位值不会影响默认内置 OCR。如确实需要继续连接单独的 PaddleX 服务，显式设置 RETAIN_LOCAL_OCR_BACKEND=service。

## 验证范围

- 使用任务 9ae55a8755c64a7a9cb06646ba619e8a 保存的 OCR 结果和译文重新排版，检查了全部三页；没有重新请求翻译模型。
- 使用 OCR_Test_1page_image_only_DE.pdf 实际运行 CPU 识别，并通过后端标准化文档校验。
- 回归测试包含原文字层、语言校验、表格字体及新增 OCR 布局、本地 OCR 路由检查。
- 当前执行环境不能直接访问 Docker Desktop 引擎；最终 Linux 镜像仍需通过上述命令构建后测试。

Paddle 的结构化文档接口参考：[官方 PP-StructureV3 文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PP-StructureV3.html)。
