# OptSpherePackGen

[English](README.md) | 简体中文

OptSpherePackGen 提供 `spherepackgen` Python 软件包，用于生成、验证、分析和导出三维微球结构，为光学散射与电磁仿真提供几何输入。

设置粒径分布、平均直径、体积分数和周期盒尺寸后，可通过本地图形界面、YAML 配置文件或 Python API 生成结构。

## 支持的结构

| 结构 | 可用选项 |
|---|---|
| 周期晶体 | 尺寸兼容的固定盒支持 SC、BCC、FCC 和金刚石立方；HCP 使用自动确定尺寸的立方体配置 |
| 不重叠随机堆积 | Poisson-disk 放置、随机顺序吸附（RSA）、力偏置松弛和原生 Lubachevsky–Stillinger（LS）压缩 |
| 允许重叠的随机介质 | 球心独立分布的标记泊松布尔球模型 |
| 粒径分布 | 单分散、准单分散、对数正态、截断正态、均匀、Gamma、Weibull、自定义粒径文件及离散混合 |

GUI 提供主要粒径分布和结构选项。离散混合、导入分布及 Metropolis 采样通过 YAML 或 Python API 配置。本版本不包含 SHU 和外部 RCPGenerator。

## 安装与启动

支持 Windows、macOS 和 Linux，需要 **Python 3.10 或以上版本**。下载或克隆仓库后，在项目目录打开终端。安装过程需要下载依赖，无需 Anaconda。

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

上述命令不需要激活虚拟环境。若希望使用下文中的简写命令，可在 Windows 执行 `.\.venv\Scripts\Activate.ps1`，或在 macOS/Linux 执行 `source .venv/bin/activate`。Windows 若限制脚本激活，继续使用上面的完整程序路径即可。

也可以安装提供的 wheel 文件：

```text
python -m pip install /path/to/spherepackgen-0.2.0-py3-none-any.whl
```

## 生成第一个结构

在 GUI 中选择粒径分布和结构类型，输入平均直径、体积分数、横向长度及深度，然后点击 **Generate**。结果包含几何验证、结构统计、坐标下载和完整结果包。

使用命令行运行：

```text
spherepackgen examples --output examples
spherepackgen run examples/fcc.yaml
```

示例随软件包安装，无需保留源码目录。若需要固定尺寸的长方体随机堆积，将以下内容保存为 `my_case.yaml`：

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
algorithm:
  name: default
analysis:
  compute_g2: true
  compute_Sk: true
  k_max_index: 3
output:
  path: results/my_case
  coordinate_unit: um
  formats: [csv, json]
  make_plots: true
  overwrite: false
runtime:
  random_seed: 12345
```

执行 `spherepackgen run my_case.yaml`。对于不重叠随机结构，自动选择规则为：`phi <= 0.18` 使用 Poisson-disk，`0.18 < phi <= 0.34` 使用 RSA，更高体积分数使用力偏置松弛。也可以手动指定其它支持的算法。

## 用于光学仿真

`particles_real_units.csv` 包含球心坐标、半径和直径，默认单位为微米。`metadata.json` 记录盒尺寸、实际体积分数、单位和验证结果。可选 HDF5 文件及 `particles_snapshot.npz` 保存无量纲几何。

将盒尺寸与球坐标导入仿真软件，创建相应球体，再在仿真软件中设置材料、折射率、波长、照明及求解器边界条件。本软件负责几何与结构统计，光学光谱由电磁求解器计算。

## 保存和复现

GUI 在选定工作目录中保存结果与历史。每次 GUI 生成使用独立目录，修改参数后仍可查看之前的结果。生成任务支持取消和时间限制。使用 `--port 8502` 更换端口，或使用 `--headless` 启动服务而不自动打开浏览器。

每次运行均记录随机种子，归档自定义粒径输入，并保存校验值、精确几何快照和 `run_bundle.zip`。解压结果包后运行：

```text
spherepackgen run /path/to/extracted/config_replay.yaml
```

复现会写入新的输出目录。若需要直接读取保存的几何：

```python
from spherepackgen import load_snapshot, run_generation

bundle = run_generation("my_case.yaml")
print(bundle.readiness)
particles = load_snapshot(bundle.config.output.path)
```

基于种子的重新生成需要相同的软件和数值依赖环境。若需要完全相同的原始坐标，应直接读取快照。

## 建模与验证说明

- 三个坐标轴均为周期边界，包括深度方向。固定盒满足 `Lx = Ly = length`、`Lz = depth`，表示重复介质。
- 固定尺寸保持不变。粒子数必须为整数，因此目标与实际体积分数可能存在小幅差异，软件会报告该差异。
- 晶体需要完整且未受应变的晶胞。GUI 可按晶胞数应用兼容尺寸；固定盒不支持 HCP。
- 重叠介质的覆盖体积分数与球体积求和得到的名义体积分数不同。报告的布尔覆盖率是模型期望值，并非球并集体积的实测结果。
- 接近堵塞密度时，收敛取决于分布、算法和计算预算。通过几何验证不能单独证明各向同性、平衡态或最大随机堵塞。
- 三维预览最多显示 1,000 个粒子；坐标导出和验证使用完整结构。

CLI 退出码 0 表示几何有效且生成器状态成功。退出码 1、2、3 分别表示输入或生成错误、验证失败、需要进一步检查的有效候选。自动化工作流使用 `--allow-candidate` 前，请阅读用户指南。

## 文档与支持

- [用户指南（简体中文）](USER_GUIDE.zh-CN.md)
- [User guide (English)](USER_GUIDE.md)
- [示例配置](configs/examples/)
- [更新记录](CHANGELOG.md)
- [贡献说明](CONTRIBUTING.md)
- [报告问题](https://github.com/Zhenpeng04/OptSpherePackGen/issues)

采用 [MIT 许可证](LICENSE)。
