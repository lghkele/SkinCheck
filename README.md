# SkinScan

Android 暗夜模式原始截图检测工具。使用 Python、OpenCV、NumPy，在 CPU 上独立运行背景异常和文本可读性检测链路。无需 OCR、神经网络、GPU、控件树或日间参考图；不缩放截图，不生成缩略图。

## 项目结构

```text
SkinScan/
├── skinscan/                         检测程序
│   ├── __init__.py                   包信息与版本号
│   ├── __main__.py                   python -m skinscan 入口
│   ├── cli.py                        命令行、输入发现、批处理、严格退出码与审核命令
│   ├── engine.py                     单图分析调度、异常隔离、问题合并、白名单与状态判定
│   ├── image.py                      原图读取、亮度／对比度、梯度与基础像素工具
│   ├── pictures.py                   图片候选定位、ROI SLICO 分割及内部／边界掩码
│   ├── background.py                 背景亮块、局部反差、细边与整页主题检测
│   ├── text.py                       文字候选生成、水平分组、局部支持与未知项分流
│   ├── textshape.py                  字缝／容器结构排除、近似中轴宽度与行结构验证
│   ├── strokes.py                    相反边缘笔画配对、方向重测、核心与双侧背景取样
│   ├── config.py                     默认阈值、配置加载／校验及页面覆盖
│   ├── models.py                     检测结果、问题和坐标关系等数据结构
│   └── report.py                     原图保存、标注／定位图、JSON 和 HTML 报告
├── docs/                             设计与实现文档
│   ├── android-dark-mode-screenshot-detection-plan.md  总体方案与算法来源
│   ├── text-algorithm.md              文字算法、测量证据与能力边界
│   ├── picture-algorithm.md           图片定位与背景／文字链路的集成
│   ├── implementation.md             实现映射、版本验收与性能记录
│   └── problem-solving-notes.md       遇到的问题、解决思路及剩余限制
├── examples/                         示例配置与合成样本
│   ├── config.yaml                   常用配置和页面覆盖示例
│   ├── all-low-contrast.yaml          全量低对比度检测配置
│   ├── generate_samples.py           原分辨率合成样本生成器
│   └── screenshots/                  生成的 PNG、元数据与批量 manifest
├── tests/                            背景、文字、笔画、图片、候选验证和 CLI 回归测试
├── reports/                          运行生成的历史报告，按运行 ID 分目录
├── pyproject.toml                    包版本、依赖、命令入口与测试配置
├── .gitignore                        本地环境、缓存、构建产物和报告的忽略规则
└── README.md                         项目入口与使用说明
```

主要调用顺序为：`cli.py` 收集输入与配置，`engine.py` 调用 `image.py` 加载原图并执行 `pictures.py` 图片定位，再并行运行 `background.py` 和 `text.py`。文字链路通过 `textshape.py` 验证候选、通过 `strokes.py` 测量笔画；引擎合并问题并判定状态，最后由 `report.py` 保存工件。

`examples/screenshots/` 由样本生成器创建，`reports/` 由扫描命令创建；本地 `.venv/`、`build/`、`dist/` 和 `*.egg-info/` 分别属于环境或构建产物，未列入核心源码结构。

## 安装与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'

python -m skinscan scan /path/to/screenshot.png --output reports
python -m skinscan scan /path/to/screenshots --config examples/config.yaml --output reports
python -m skinscan scan --manifest /path/to/manifest.json --workers 2 --output reports
python -m pytest -q
```

要求 Python 3.9+，依赖与版本范围见 `pyproject.toml`。只接受 **8 位、不透明、无损 PNG**，支持灰度 PNG。JPEG、损坏图片、16 位 PNG、含透明像素的 PNG 都返回错误。输入目录递归发现 `.png`，不修改源文件；输出目录应放在输入截图目录之外。

每次运行生成独立的 `reports/<UTC时间>-<随机ID>/`，保存 `summary.json`、`index.html` 和各截图的 `original.png`、`annotated.png`、`result.json`、`review.json`。原图逐字节复制，标注图与原图尺寸相同。HTML 仅用 CSS 适配屏幕显示，处理过程始终使用完整像素。点击图片可以查看完整图。

## 输入元数据

每张截图可以有同名 JSON，例如 `home.png` 配 `home.json`：

```json
{
  "pageId": "home",
  "caseId": "home_default",
  "theme": "dark",
  "width": 1080,
  "height": 2400,
  "device": "Pixel",
  "androidVersion": "16",
  "capturedAt": "2026-10-19T09:00:00Z"
}
```

不提供时，使用文件名作为 `pageId`，默认预期暗夜模式。传入的宽高必须与 PNG 一致；主题必须为 `dark`。也可对单张截图指定 `--metadata metadata.json`。

批量 manifest 是数组，或 `{"screenshots": [...]}`；每项包括 `imagePath` 和上述可选字段。相对路径基于 manifest 所在目录解析。manifest 元数据优先于同名 sidecar JSON。无效的单张图片或 sidecar 不会中止其他截图；manifest 结构错误会在运行前报错。

Android 端通过已有自动化进入业务状态、等待动画结束后采集，例如：

```bash
adb exec-out screencap -p > home.png
```

请由测试用例确保应用已切换暗夜模式，并固定字体和显示缩放；此程序只消费最终截图。

## 图片内容与误报过滤

图片定位自 0.3.0 起提供：学习平坦底色生成候选，结合局部颜色／Sobel 细节分布、边界和重复布局筛选，再对候选原图区域执行 SLICO 超像素分割。仅高可信且有足够分割支持的图片内部参与背景过滤；边界和不确定区域保留复核，0.4.0 起文字链路也使用图片范围加强内部候选验证，稳定标签仍逐笔画检测。细边使用原始阈值连通结构，避免图片掩码制造假边框。

每张结果记录 `statistics.pictureRegions`、图片内部像素数、原图与剩余 UI 的亮度统计。图片区域不会加入 `ignored_regions`。可定位图片时额外输出原尺寸 `pictures.png`：绿色为采用的图片范围，橙色为未自动排除的候选；实际排除掩码在调试模式的 `pictures_interiors.png` 中。图片外接框不等于所有像素都被排除。

配置 `pictures.enabled: false` 可以关闭此分支。默认不启用 LBP，不理解封面语义，也不能可靠区分纯色海报和普通 UI 卡片。算法与论文关系见 [图片算法说明](docs/picture-algorithm.md)。

依赖已由 `opencv-python-headless` 改为 `opencv-contrib-python-headless`，以提供 CPU SLICO。升级已有虚拟环境时先卸载 `opencv-python-headless`，再安装本项目；同一环境不要混装多个提供 `cv2` 的 OpenCV 包。

## 算法与配置

总体设计、各处理环节的算法与论文来源见[总体方案](docs/android-dark-mode-screenshot-detection-plan.md)。其中明确区分论文方法、借鉴思想和项目自定义规则。

从白卡灰字、整行取色到封面误报、字缝和圆角过滤的演进、验证证据及未解决问题，见[遇到的问题与解决思路](docs/problem-solving-notes.md)。

- 背景：绝对亮度、页面中位数偏移和多阈值局部环境反差分别生成候选；计算连通面积、矩形度、亮度均匀度、外围中位数和纹理。窄边从未经开运算的掩码独立提取，支持单像素环和三边结构，默认交人工确认。全页规则综合高亮/暗色比例与最大低纹理亮区。
- 文本：多尺度 Black-hat/Top-hat 加自适应噪声阈值和 Sobel 弱边缘；按尺寸、笔画填充、相近高度和水平间距组合。借鉴 SWT 的相反梯度边缘配对思想，按局部笔画宽度采样核心及两侧邻近背景；拒绝颜色不稳定、两侧背景不一致或经过忽略像素的配对。逐候选组件统计局部低对比度支持，再组合实际受影响的文字段，避免整行中的深色字掩盖浅灰字。组件可能是偏旁或粘连字符，不等于完整单字；此流程没有实现完整 SWT 检测器。
- 文本范围：默认 `text.detection_mode: theme_mismatch`，仅检测暗笔画配深色局部背景、亮笔画配浅色局部背景；白卡灰字不进入专项文字异常分支。可用 `all_low_contrast` 检查全部低对比度组合，见 `examples/all-low-contrast.yaml`。背景亮区检测保持独立，白卡本身仍可触发背景异常。
- 对比度：**背景筛选亮度**统一为 `Y=(0.2126R+0.7152G+0.0722B)/255`，**文字对比度**使用线性化 sRGB 和 WCAG 公式。两套阈值不混用，实际配置随每张结果保存。
- 过滤：以周围同色比例、同色连通性、相反极性响应和近似中轴宽度排除字缝、容器边缘及不规则组件；图片内部要求更多配对和空间支持。普通单字／数字仍测量，图片上稳定底色的标签也保留。采样失败仍可能漏检，候选超限记录分析不完整。
- 合并：重复候选保留全部规则和其他检测证据，不把相关阈值的重复命中当成独立概率。背景包含文本时以父子问题展示；CI 检查所有层级的问题。

全部可配置参数和默认值集中在 `skinscan/config.py`，示例见 `examples/config.yaml`。不认识的配置键、无效阈值和不合法坐标会被拒绝。坐标均为原始像素的 **`[left, top, right, bottom)`**；右、下边界不包含。超出图像的忽略坐标会裁剪，不会改变检测分辨率。

`theme_mismatch` 使用的局部背景亮度范围默认为深色 `Y <= 0.30`、浅色 `Y >= 0.75`，中间范围不参与该专项检测；两者都可按页面配置。其目标是发现主题颜色错配，并不覆盖所有文字可读性问题。采样参数位于 `text.stroke_sampling`。无法建立足够笔画配对时，最多尝试两个邻近方向；仍失败的候选只有在形成较强 UI 文字结构时才输出无法测量，其余计入诊断。0.4.0 取消整行估色告警。JSON 的 `measurementMethod` 区分 `paired_strokes` 与 `unmeasured_text_structure`，后者没有估算对比度。`strokePairs` 保存有限数量的实际配对及原图坐标。算法细节见 `docs/text-algorithm.md`。

`global.ignored_regions` 和页面的 `ignored_regions` 合并，忽略像素不参与亮度统计、候选和颜色采样。`pages.<pageId>` 可覆盖 `pictures`、`background`、`text`、`filter`、`scoring` 和 `runtime`；`ignore_page: true` 跳过整页。页面 `runtime` 的 `save_debug_masks` 可以覆盖全局，批量并发和是否生成汇总 HTML 则由全局控制。`allowed_colors` 只降低背景风险，记录命中并保留告警，避免放过大面积错误背景，也不会掩盖该颜色上的低对比度文字。

模板匹配白名单为设计中的可选能力，当前未实现。

## 状态与 CI

严重程度与置信度分别输出。置信度是规则评分，不是统计概率；真实样本校准前不能解读成准确率。

| 检测状态 | 含义 |
| --- | --- |
| `passed` | 已执行的规则未发现问题 |
| `failed` | 发现达到阈值的高置信度严重/高风险问题 |
| `warning` | 检测到中风险提示；可附带人工复核要求，不阻塞 CI |
| `uncertain` | 有低置信度问题、估算不稳定或候选超限，需人工审核 |
| `ignored` | 页面白名单或全部像素被忽略，不计为通过 |
| `error` | 输入、配置或预处理错误，无法检测 |

同一张图可以同时包含已确定风险与需要复核的问题；`status` 可以是 `failed`，而 `reviewStatus` 仍为 `needs_review`。某条检测链路失败时，另一条继续运行；`modules` 和 `errors` 明确记录不完整分析，不能按通过处理。

默认报告模式：算法告警不阻塞 CI；输入或模块错误仍返回 `2`。使用 `--strict` 启用阻断：

| 退出码 | 含义 |
| ---: | --- |
| `0` | 运行完成；默认模式允许算法告警；严格模式允许通过、中风险提示和明确忽略 |
| `1` | 严格模式发现高置信度、无需补充审核的严重/高风险问题 |
| `2` | 输入/配置/运行错误，或严格模式存在无法可靠判断的分析 |

严重程度阈值默认为对比度 `<2` 严重、`<3` 高风险、`<4.5` 中风险。置信度默认 `0.80`；窄边、短文本和不稳定背景不会直接被确认为阻断风险。配置 `scoring.fail_severities` 可以调整阻断级别。

```bash
python -m skinscan scan --manifest screenshots/manifest.json --strict --output reports
```

流水线归档整个运行目录即可保留历史。没有网络服务、数据库或自动删除历史的行为。

## 人工复核

```bash
python -m skinscan review reports/<run>/<image>/result.json \
  --issue <result.json中的issue.id> --verdict dismissed --reviewer qa --note '允许的品牌元素'

python -m skinscan review reports/<run>/<image> \
  --verdict confirmed --reviewer qa --note '已确认未适配'
```

`confirmed` / `dismissed` / `unresolved` 保存至同目录的 `review.json`，记录审核者、备注、UTC 时间和每次修改历史。HTML 有审核记录链接。复核不覆盖机器 JSON，不自动更改阈值、白名单或历史 CI 判定；可把确认的误报整理为后续校准样本。

## 验证与能力边界

生成与运行 1080×2400 合成验收样本：

```bash
python examples/generate_samples.py --output examples/screenshots
python -m skinscan scan --manifest examples/screenshots/manifest.json --config examples/config.yaml --workers 2 --output reports
# 全量低对比度检测，白卡灰字仍会按真实对比度命中：
python -m skinscan scan --manifest examples/screenshots/manifest.json --config examples/all-low-contrast.yaml --output reports
```

当前 15 张合成样本包含白色天空封面、白卡嵌图及图片旁弱文字；99 项测试通过。测试覆盖方案两类核心颜色组合、独立局部反差、整页亮色、正常白字、窄边/开口边缘、单字/数字/短文本、渐变/复杂纹理、白名单、模块异常隔离、输入错误、原图保真、历史报告、审核反馈和 CI 策略。更多实现/验收说明见 `docs/implementation.md`。

合成样本用于验证机制，不代表真实页面的召回率和误报率。当前有用户提供的单张封面页验证示例，但没有方案中两个案例的原始 PNG，也没有人工标注的业务截图集；仍需用实际页面校准阈值，并分别统计长文本/短文本、窄边/正常边框、背景召回率、区域准确率和人工复核量。纯截图无法发现与背景完全同色、没有任何可见轮廓的文字；图标、海报、二维码、设计允许的弱文字和复杂背景需要白名单或复核。此工具不能替代设计验收。
