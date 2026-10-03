# OptSpherePackGen 使用指南

[English](USER_GUIDE.md) | 简体中文 · [README](README.zh-CN.md)

本指南介绍如何生成微球结构、检查几何质量，并将结果用于光学仿真。
Python 包和命令行程序的名称均为 `spherepackgen`。

## 1. 安装与工作目录

使用 Python 3.10 或更新版本。下载或克隆仓库后，在项目目录中打开终端，
创建虚拟环境并安装。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\spherepackgen.exe gui --workdir .
```

macOS / Linux：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install .
./.venv/bin/spherepackgen gui --workdir .
```

也可以安装 wheel 文件：
`python -m pip install /path/to/spherepackgen-0.2.0-py3-none-any.whl`。
安装时需要下载依赖，无需使用 Anaconda。

上述命令无需激活虚拟环境。使用下文较短的 `spherepackgen` 命令前，
Windows 用户可运行 `.\.venv\Scripts\Activate.ps1`，
macOS/Linux 用户可运行 `source .venv/bin/activate`。
如果 Windows 阻止执行激活脚本，直接使用 `spherepackgen.exe` 的完整路径即可。

为配置、上传文件和生成结果选择一个可写目录：

```text
spherepackgen gui --workdir /path/to/workspace
spherepackgen examples --output examples
spherepackgen run examples/fcc.yaml
```

未指定 `--workdir` 时，GUI 使用启动时的目录。
上传缓存和历史记录位于工作目录的 `.spherepackgen/` 中；
再次使用同一工作目录启动，可查看已保存的历史记录。
导出示例时不会覆盖已有的示例文件。

在源码目录执行 `python -m pip install ".[visualization]"` 可安装可选的
PyVista 渲染依赖；未安装时仍可使用 matplotlib 渲染。

## 2. 选择结构与粒径分布

| 空间排列 | 几何特征 | 常用选择 |
|---|---|---|
| `periodic_crystal` | 重复排列的晶体晶胞 | SC、BCC、FCC、金刚石立方；HCP 使用自动确定尺寸的立方盒 |
| `hard_core_random` | 满足周期边界下不重叠条件的随机微球 | Poisson-disk、RSA、force-biased、LS |
| `overlapping_random` | 球心独立随机分布，允许微球重叠 | 标记泊松 Boolean 介质 |

| 粒径类别 | 配置方式 |
|---|---|
| `monodisperse` | 所有微球半径相同 |
| `quasi_monodisperse` | 截断正态采样；`CV_radius` 默认为 0.03 |
| `continuous_polydisperse` | 选择 `lognormal`、`truncated_normal`、`uniform`、`gamma`、`weibull` 或 `custom_file` |
| `discrete_mixture` | 通过 YAML/API 设置相对粒径与数量比例 |
| `imported_distribution` | 通过 YAML/API 从粒径文件采样 |

GUI 提供单分散、准单分散和连续多分散选项，晶体结构要求使用单分散微球。
当前不提供 SHU、外部 RCPGenerator、通用超均匀结构、准晶体或自定义关联函数结构生成。

## 3. 设置物理尺寸与体积分数

内部长度无量纲化，使平均粒径等于 1。
物理长度输入支持 `200 nm`、`0.2 um` 等带单位的写法。
实际坐标默认以 `um` 导出。

| 字段 | 含义 |
|---|---|
| `physical.mean_diameter` | 生成样本的平均直径 |
| `physical.packing_fraction` | 不重叠结构的目标微球体积分数 |
| `physical.covered_volume_fraction` | 重叠介质的目标期望覆盖率，取值介于 0 与 1 之间 |
| `domain.length` | 固定横向长度，x 与 y 共用 |
| `domain.depth` | 固定 z 向长度 |
| `particles.num_particles` | `auto` 或正整数 |
| `runtime.random_seed` | 非负整数，或 `null`，由程序生成并记录种子 |

使用 `domain.type: periodic_box` 时，必须同时提供长度和深度。
盒尺寸为 `(L, L, D)`，**三个方向均采用周期边界**。
不要同时设置 `physical.medium_thickness` 或 `domain.box_size`。

自动选择粒子数时，程序保持设定的平均粒径和固定盒尺寸，
根据采样微球的体积选择整数粒子数。等粒径微球近似满足：

```text
N ≈ round(phi × L² × D / (pi × mean_diameter³ / 6))
```

整数粒子数可能造成小幅体积分数偏差。报告会区分粒子数离散化允许的偏差和验证容差；
前者为最大采样微球体积的一半除以盒体积。
手动粒子数与目标参数冲突，或微球与自身周期镜像重叠时，程序会报错。

无需固定长度和深度的立方盒可使用 `domain.type: periodic_cube`。
程序可根据粒子数和体积分数推算边长，也可使用
`physical.medium_thickness` 作为近似目标边长。
由于粒子数和晶胞数是离散的，最终边长可能与输入值略有不同。

### 完整的随机堆积配置

保存为 `my_case.yaml`：

```yaml
project:
  name: my_case
structure:
  size_distribution: monodisperse
  spatial_order: hard_core_random
physical:
  mean_diameter: 200 nm
  packing_fraction: 0.15
domain:
  type: periodic_box
  length: 1.2 um
  depth: 0.6 um
particles:
  num_particles: auto
size_distribution:
  type: monodisperse
algorithm:
  name: default
analysis:
  compute_g2: true
  compute_Sk: true
  bins: 80
  k_max_index: 3
validation:
  require_non_overlap: true
  tolerance_phi: 1.0e-6
  tolerance_overlap: 1.0e-5
output:
  path: results/my_case
  coordinate_unit: um
  formats: [csv, json]
  make_plots: true
  overwrite: false
runtime:
  random_seed: 12345
```

运行 `spherepackgen run my_case.yaml`。

### 可重叠的 Boolean 介质

在上述完整配置中，将结构与物理参数替换为下列内容，
并设置 `validation.require_non_overlap: false`：

```yaml
structure:
  size_distribution: monodisperse
  spatial_order: overlapping_random
physical:
  mean_diameter: 200 nm
  covered_volume_fraction: 0.70
```

独立球心模型中，覆盖率 `c` 与名义体积分数 `eta` 满足
`eta = -ln(1 - c)`。
名义体积分数等于微球体积之和除以盒体积；重叠区域会重复计入，因此可以大于 1。
报告中的 `1 - exp(-eta)` 是期望覆盖率，
并非对当前有限样本中所有微球并集体积的直接测量。

## 4. 设置粒径

使用连续分布时，同时修改结构的粒径类别和粒径分布配置：

```yaml
structure:
  size_distribution: continuous_polydisperse
  spatial_order: hard_core_random
size_distribution:
  type: continuous_polydisperse
  distribution: lognormal
  CV_radius: 0.20
  min_radius_factor: 0.40
  max_radius_factor: 1.80
```

| 分布 | 参数 |
|---|---|
| `lognormal` | `CV_radius`；可选的半径倍数上下限 |
| `truncated_normal` | `CV_radius` 与半径倍数上下限 |
| `uniform` | 半径倍数的下限与上限 |
| `gamma` | `CV_radius` 或 `parameters.shape` |
| `weibull` | `parameters.shape` |
| `custom_file` | `size_file`、`file_values`，以及可选的 `file_column` |

半径倍数相对于名义平均半径。采样或截断后，程序还会归一化，
使样本满足指定的平均直径，因此最终粒径边界和实测 CV 可能与输入设置不同。
对边界或分布宽度有要求时，应检查导出的粒径统计。

### 自定义粒径文件

对 `continuous_polydisperse` 使用以下配置：

```yaml
size_distribution:
  type: continuous_polydisperse
  distribution: custom_file
  size_file: particle_sizes.csv
  file_values: diameter
  file_column: diameter_um
```

CSV 示例：

```csv
diameter_um
0.20
0.20
0.40
0.40
```

支持 CSV、TXT 或 DAT 文件，数值可用逗号或空白分隔。
请提供由有限正数构成的干净数值列。
可用表头名称选择列，或使用从 0 开始的整数列索引；
省略 `file_column` 时，使用第一个数值列。
文件路径相对于 YAML 配置文件解析。

文件提供的是用于采样的粒径总体。当所需粒子数超过输入行数时，采用有放回采样。
生成的样本随后归一化到 `physical.mean_diameter`。
输入行并非逐粒子固定清单，也不会严格保证各粒径组的数量。
程序不会从列名或粒径值自动推断材料映射。

在 GUI 中选择 **Continuous distribution → custom_file**，
上传文件，并指定其中的数值代表直径还是半径。

### 通过 YAML 设置离散混合粒径

设置 `structure.size_distribution: discrete_mixture`，并使用：

```yaml
size_distribution:
  type: discrete_mixture
  species:
    - radius_factor: 1.0
      number_fraction: 0.5
    - radius_factor: 2.0
      number_fraction: 0.5
```

半径倍数指定相对粒径，数量比例作为采样概率，不保证精确配额。
所有采样半径会一起归一化到指定平均直径。
这些参数定义几何尺寸，不定义光学材料。

## 5. 选择生成算法

| 算法 | 行为 |
|---|---|
| `crystal` | 确定性地重复排列晶胞 |
| `marked_poisson_boolean` | 独立随机球心，允许微球重叠 |
| `poisson_disk` | 通过局部不重叠检查放置可变半径微球 |
| `rsa` | 随机顺序插入，直到全部放置或尝试次数耗尽 |
| `force_biased` | 通过生长和排斥松弛达到目标粒径 |
| `lubachevsky_stillinger` | 半径生长过程中的事件驱动运动与弹性碰撞 |
| `metropolis` | RSA 初始化后尝试随机移动，仅接受不重叠的移动；通过 YAML/API 使用 |

设置 `algorithm.name: default` 时：

| 空间排列 / 体积分数 | 选择的算法 |
|---|---|
| `periodic_crystal` | `crystal` |
| `overlapping_random` | `marked_poisson_boolean` |
| `hard_core_random`，`phi <= 0.18` | `poisson_disk` |
| `hard_core_random`，`0.18 < phi <= 0.34` | `rsa` |
| `hard_core_random`，`phi > 0.34` | `force_biased` |

自动选择的 Poisson-disk 或 RSA 放置失败时，流程可以转用 force-biased。
显式选择 Poisson-disk 或 RSA 时，程序保持该选择并报告放置失败。

force-biased 参数包括 `initial_radius_fraction`、`stages`、
`relaxation_steps_per_stage` 和 `contraction_rate`。
当 `phi >= 0.60` 时，默认使用更大的松弛和清理预算。
LS 参数包括 `initialization`、`initial_radius_scale`、
`compression_rate`、`velocity_scale`、`max_events`、
`event_backend` 和 `final_cleanup_steps`。
LS 的自动初始化在 `phi >= 0.60` 时选择 force-biased 热启动。
完整参数可参考 [force-biased 示例](configs/examples/force_biased.yaml)
和 [LS 示例](configs/examples/lubachevsky_stillinger.yaml)。

更高的体积分数或更宽的粒径分布可能需要更长运行时间和参数调整。
仅凭 force-biased 或 LS 输出，不能证明结构达到平衡、
随机密堆积或最大随机堵塞状态。

### 晶体结构

以下配置生成尺寸自动确定的立方盒晶体：

```yaml
project:
  name: fcc_case
structure:
  size_distribution: monodisperse
  spatial_order: periodic_crystal
physical:
  mean_diameter: 200 nm
  packing_fraction: 0.50
domain:
  type: periodic_cube
particles:
  num_particles: auto
algorithm:
  name: crystal
  parameters:
    lattice_type: FCC
    unit_cells: 2
output:
  path: results/fcc_case
  make_plots: false
  overwrite: false
```

固定盒需要完整、无应变的晶胞。
可以使用三元素的 `unit_cells: [nx, ny, nz]` 指定各方向的晶胞数，
但每个晶胞的边长必须一致，且所得体积分数必须与目标值一致。
GUI 的 **Apply compatible crystal dimensions** 按钮根据晶胞数计算相容的长度和深度。
固定尺寸盒不支持 HCP。

不重叠晶格的最大体积分数近似为：
SC 0.5236、BCC 0.6802、FCC/HCP 0.7405、金刚石立方 0.3401。
改变随机种子不会改变确定性的晶体结构。

## 6. 使用 GUI

运行 `spherepackgen gui --workdir /path/to/workspace`。
服务绑定到本机 localhost。
默认端口被占用时可增加 `--port 8502`；
使用 `--headless` 可跳过自动打开浏览器。

1. 在 **Run** 中选择粒径分布、结构、尺寸和体积分数。
2. 查看粒子数预览与输入错误提示。
3. 在 **Advanced** 中调整分布、算法、分析、验证和导出设置。
   配置预览会反映当前设置。
4. 点击 **Generate**，查看各阶段进度。
5. 在 **Results** 中检查验证结果并下载文件；在 **History** 中查看历史运行。

GUI 将解析后的粒子数限制为 100,000，默认运行时间上限为 600 秒。
点击 **Cancel run** 或达到超时时间会停止后台任务，并保留此前完成的结果。
每次生成使用独立的输出目录，页面只显示该次运行清单列出的文件。

Quick、Standard 和 Detailed 三种 `S(k)` 预设使用的最大倒空间索引分别为
5、10 和 20。增大该索引会显著增加计算量。
关闭绘图不会关闭分析；需分别使用 `Compute g2(r)` 和 `Compute S(k)` 控件控制分析。

GUI 默认导出 CSV/JSON、启用绘图，并生成一张透视堆积视图。HDF5 为可选项。
预览图最多显示 1,000 个粒子，并注明显示的是子集；
导出几何和验证覆盖全部粒子。

## 7. 理解验证与输出

程序检查坐标是否有限、半径是否为正、周期边界、体积分数一致性、
粒径统计以及所要求的不重叠条件。
用于仿真前，请检查错误、警告、生成器诊断和 `export_ready` 标记。

| CLI 退出码 | 含义 |
|---|---|
| 0 | 几何有效且生成器成功；或使用 `--allow-candidate` 显式接受了几何有效的候选结果 |
| 1 | 输入、文件或生成错误 |
| 2 | 几何验证失败 |
| 3 | 几何有效，但生成器有警告或未达到目标 |

默认采用严格判定。
`spherepackgen run my_case.yaml --allow-candidate` 改变批处理任务对有效候选结果的接受方式，
不会修改记录中的就绪标记，也不会放行几何验证失败的结果。

`tolerance_phi` 是无量纲体积分数的绝对容差；
`tolerance_overlap` 是内部无量纲长度单位下的重叠容差。
增大容差不能修复无效几何。

| 文件 | 内容 |
|---|---|
| `particles_real_units.csv` | 使用 `output.coordinate_unit`（默认 `um`）的球心、半径和直径 |
| `particles_dimensionless.csv` | 平均直径归一化为 1 的几何 |
| `metadata.json` | 尺寸、单位、解析后参数、诊断、验证与就绪状态 |
| `validation_report.json` | 详细验证指标 |
| `config_input.yaml` / `config_resolved.yaml` | 原始输入与解析后配置 |
| `config_replay.yaml` | 有效种子、归档输入路径和新的输出位置 |
| `particles_snapshot.npz` | 精确的无量纲几何与标签 |
| `inputs/*` | 归档的自定义粒径输入 |
| `environment.json` | 软件/源码信息和依赖版本 |
| `run_manifest.json` | 输出清单、校验和与就绪状态 |
| `run_bundle.zip` | 可迁移的配置、输入和结果包 |
| `analysis/*.csv` | 启用分析时的最近邻距离、`g2(r)` 和 `S(k)` |
| `particles.h5` | 可选的无量纲几何和分析 |
| `figures/*.png` | 启用绘图时的粒径直方图、分析图和堆积预览 |
| `logs/run.log` | 运行摘要 |

坐标 CSV 的列为
`particle_id,x,y,z,radius,diameter,species_id,material_id`。
请使用导出的实际直径，不要假定每个微球都等于平均粒径。
标签字段不包含折射率或材料数据库；生成的材料 ID 默认为零。

元数据同时记录无量纲和米制盒尺寸。
HDF5 几何、`g2` 距离和最近邻距离使用内部长度；
`S(k)` 波矢使用内部长度的倒数。
长度乘以物理平均直径即可转换，波矢则需除以物理平均直径。
长方盒的 `g2(r)` 范围到最短边长的一半。

相对输出路径默认以启动目录为基准。
设置 `output.relative_to_config: true` 可改为以 YAML 所在目录为基准。
CLI 默认使用 `output.overwrite: true`；
选择 `false` 可保留已有结果并创建新的目录。GUI 已为每次运行隔离输出。

## 8. 迁移与复现结构

使用 **Download complete result bundle** 下载完整结果包。
在另一台已安装本软件的电脑上解压，然后运行：

```text
spherepackgen run /path/to/extracted/config_replay.yaml
```

粒径输入文件随结果包一起保存，并通过校验和验证。
重放结果写入配置旁的 `replay/`；必要时添加唯一后缀，保留原始结果。

种子为 `null` 时，程序为该次运行生成并记录种子。
再次运行原始的无种子配置会得到另一个样本；
复现原运行时，应使用记录了有效设置的重放配置。
基于种子的重新生成要求软件和数值依赖匹配；
不同平台上的优化轨迹不保证完全一致。

无需重新生成即可准确加载原始坐标：

```python
from spherepackgen import load_snapshot, run_generation

bundle = run_generation("my_case.yaml")
print(bundle.readiness)
particles = load_snapshot(bundle.config.output.path)
centers_um = particles.positions * (bundle.resolved.mean_diameter_m / 1e-6)
radii_um = particles.radii * (bundle.resolved.mean_diameter_m / 1e-6)
```

`load_snapshot` 验证快照校验和。在显式转换前，坐标和半径仍为无量纲值。

## 9. 对接光学仿真软件

导入 `particles_real_units.csv`，并读取 `metadata.json` 中的盒尺寸。
以每个球心 `(x, y, z)` 和对应 `radius` 建立微球。
在仿真软件中分配材料和光学参数，包括按粒径分配材料。
自定义粒径文件不会自动执行这一步。

周期微球可能跨越盒面。在仿真软件中表示对应的周期镜像或适当裁剪，
不要因为球面超出盒子就删除该微球。
生成结构在深度方向也采用周期边界。
若要模拟入口和出口开放的有限厚度薄层，需要显式构建薄层及其边界，
并检查所得界面。

几何有效性和结构统计不能直接确定光学响应。
还需在光学求解器中设置网格、入射条件、波长和边界，并检查数值收敛。

## 10. 常见问题

| 现象 | 处理方式 |
|---|---|
| 找不到 `spherepackgen` 命令 | 激活虚拟环境，或使用程序完整路径 |
| Windows 阻止激活 | 直接运行 `.\.venv\Scripts\spherepackgen.exe` |
| 端口被占用 | 启动时增加 `--port 8502` |
| 粒子数与体积分数冲突 | 选择 `auto`，或统一粒子数、粒径、盒尺寸和体积分数 |
| 与自身周期镜像重叠 | 增大最短盒尺寸或减小粒径 |
| 晶体尺寸不相容 | 在 GUI 中应用相容晶体尺寸 |
| 放置失败或生成超时 | 检查体积分数与粒径分布，选择其他算法或增加相应预算 |
| 缺少分析文件 | 启用对应的计算；绘图设置独立控制 |
| 自定义粒径与输入数值不同 | 检查采样、`file_values` 和指定平均值的归一化 |
| 重放校验和失败 | 重新解压完整结果包，避免修改归档输入 |

无法解决时，可提交 [GitHub issue](https://github.com/Zhenpeng04/OptSpherePackGen/issues)，
附上配置、种子、版本、操作系统和错误/验证报告。
分享结果包前，请检查是否含有私人路径或数据。

另见 [README](README.zh-CN.md)、[示例配置](configs/examples/) 和 [MIT 许可证](LICENSE)。
