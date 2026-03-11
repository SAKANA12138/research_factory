<div align="center">

# 🏭 Research Factory RF-1.5

### 多智能体 AI 科研自动化流水线

*从热点发现 → 源码解剖 → 对抗辩论 → 论文生成，全程 Agent 驱动*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub Models](https://img.shields.io/badge/LLM-GitHub%20Models%20API-181717?style=for-the-badge&logo=github&logoColor=white)](https://models.github.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Agents](https://img.shields.io/badge/Agents-12-FF6F00?style=for-the-badge&logo=openai&logoColor=white)](#-agent-角色全览)

<br/>

<img src="https://img.shields.io/badge/Phase%201-Intelligence-blue?style=flat-square" alt="Phase 1"/>
→
<img src="https://img.shields.io/badge/Phase%202-Dissection-purple?style=flat-square" alt="Phase 2"/>
→
<img src="https://img.shields.io/badge/Phase%203-Debate-red?style=flat-square" alt="Phase 3"/>
→
<img src="https://img.shields.io/badge/Phase%204-Publication-brightgreen?style=flat-square" alt="Phase 4"/>

</div>

---

## 📑 目录

- [🌟 项目简介](#-项目简介)
- [🧠 系统架构](#-系统架构)
- [🤖 Agent 角色全览](#-agent-角色全览)
- [⚡ 快速开始](#-快速开始)
- [🔧 配置说明](#-配置说明)
- [🚀 使用方式](#-使用方式)
- [📂 项目结构](#-项目结构)
- [📊 输出产物](#-输出产物)
- [🔬 四阶段流水线详解](#-四阶段流水线详解)
- [🛠️ 开发与测试](#️-开发与测试)
- [❓ 常见问题](#-常见问题)
- [🗺️ 路线图](#️-路线图)
- [📄 开源协议](#-开源协议)

---

## 🌟 项目简介

**Research Factory (RF-1.5)** 是一个工业级多智能体科研自动化系统。它编排 **12 个专业化 AI Agent**，通过 **4 个级联阶段**，自动完成从前沿论文调研到 LaTeX 论文初稿生成的全流程：

```
输入: 一个研究主题（如 "将线性注意力算子嫁接到 Transformer LLM"）
        ↓
🔍 Phase 1  全网情报感知（arXiv / GitHub / OpenReview 数据采集 + 热度/显存/创新度分析）
        ↓
🔬 Phase 2  深度解剖建模（源码张量流向提取 + 数学公式 → PyTorch 代码 + 冲突检测）
        ↓
⚔️  Phase 3  对抗博弈辩论（激进派提案 → 保守派攻击 → 调解员融合最终方案）
        ↓
📝 Phase 4  质量自检投稿（多维度评分 + 投稿收益建模 + LaTeX 自动编译）
        ↓
输出: 完整 LaTeX 论文源码 + PDF + 结构化研究报告 JSON
```

### 核心特性

| 特性 | 说明 |
|------|------|
| 🤖 **12 Agent 协同** | 每个 Agent 有独立系统提示词、专属职能、结构化 JSON 输出 |
| 🔗 **Context 瀑布流** | 上游 Agent 产出自动注入下游，信息零损耗传递 |
| 🌐 **真实数据源** | Phase 1 实际调用 arXiv API、GitHub REST API、OpenReview API |
| ⚔️ **对抗博弈机制** | Proposer → Critic → Mediator 三轮辩论，模拟真实 Peer Review |
| 📐 **数学→代码翻译** | 自动将嫁接点公式转化为对齐的 PyTorch `nn.Module` |
| 📄 **端到端论文生成** | 从标题到参考文献的完整 LaTeX 源码 + PDF 编译 |
| 🔌 **多模型支持** | GPT-4.1 / GPT-4o / o4-mini / Claude 等，一行配置切换 |
| 💾 **显存红线审计** | 内置 VRAM 分项估算（参数/优化器/激活值/KV-Cache） |

---

## 🧠 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RF-1.5 Pipeline Orchestrator                 │
│                           (rf/main.py)                              │
├─────────────┬──────────────┬───────────────┬────────────────────────┤
│  Phase 1 🔍 │  Phase 2 🔬  │  Phase 3 ⚔️   │  Phase 4 📝            │
│             │              │               │                        │
│ TrendAnalyst│ SrcArchaeo-  │  Proposer     │ QualityInspector       │
│ HW Guardian │  logist      │  Critic       │ PublicationStrategist  │
│ NoveltyAudt │ MathTransltr │  Mediator     │ EditorInChief          │
│             │ ConflictDtcr │               │                        │
├─────────────┴──────────────┴───────────────┴────────────────────────┤
│                      Shared LLM Client (rf/llm/)                    │
│            GitHub Models API  ·  OpenAI-Compatible Endpoint         │
│     https://models.github.ai/inference → GPT-4.1 / Claude / ...    │
├─────────────────────────────────────────────────────────────────────┤
│                      External Tools (rf/tools/)                     │
│         arXiv API  ·  GitHub REST API  ·  OpenReview API v2         │
│                     LaTeX Compiler (pdflatex)                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent 角色全览

### Phase 1 — 全网情报感知 (Intelligence)

| Agent | 角色名 | 职责 | 输出 Key |
|-------|--------|------|----------|
| 🔍 | **热度分析师** (Trend Analyst) | 捕捉技术爆发点：扫描 arXiv / GitHub / OpenReview，分析社交热度、Star 增速、审稿评分 | `trend_report` |
| 🖥️ | **可行性审计师** (Hardware Guardian) | 显存红线审查：分项估算 VRAM 占用（参数 + 优化器 + 激活值 + KV-Cache），评估 INT8 量化 + LoRA 可行性 | `hardware_audit` |
| 🔎 | **查重比对官** (Novelty Auditor) | 创新度排查：对比近两年论文的核心算子与实验路径，标注相似度并建议差异化方向 | `novelty_audit` |

### Phase 2 — 深度解剖建模 (Dissection)

| Agent | 角色名 | 职责 | 输出 Key |
|-------|--------|------|----------|
| 🏛️ | **源码考古学家** (Source Archaeologist) | 提取 `modeling_*.py` 中的张量流向图，定位可嫁接子模块及其边界 tensor shape | `source_analysis` |
| 📐 | **数学翻译官** (Math Translator) | 将嫁接点的数学公式转化为 PyTorch `nn.Module`，附 shape 对齐证明 | `math_translation` |
| ⚡ | **冲突检测员** (Conflict Detector) | 识别 Dim / Norm / PE 三类排异风险，输出兼容性评分和集成清单 | `conflict_report` |

### Phase 3 — 对抗博弈 (Debate)

| Agent | 角色名 | 职责 | 输出 Key |
|-------|--------|------|----------|
| 🚀 | **激进派** (Proposer) | 提出完整嫁接方案：策略 + 训练配方 + 完整代码 + Benchmark 预测 | `proposal` |
| 👿 | **保守派** (Critic) | 模拟最严苛审稿人：攻击数学稳定性、显存溢出、实验缺陷、创新不足 | `criticism` |
| 🤝 | **调解员** (Mediator) | 融合双方意见，产出最终可执行实验方案（含 ablation 设计 + GO/NO-GO 决策） | `final_plan` |

### Phase 4 — 质量自检与投稿 (Quality)

| Agent | 角色名 | 职责 | 输出 Key |
|-------|--------|------|----------|
| ✅ | **质量自检员** (Quality Inspector) | 多维度自评：逻辑严密性 / 实验完备性 / 图表专业度 / 写作质量，各 25 分 | `quality_report` |
| 🎯 | **投稿战略官** (Publication Strategist) | 期望收益建模：$E = P(\text{accept}) \times \text{Impact}$，输出"保底+冲击"投稿组合 | `publication_strategy` |
| 📄 | **总编辑** (Editor-in-Chief) | 将全部成果编译为完整 LaTeX 源码（Title → References），尝试 PDF 编译 | `paper_draft` |

---

## ⚡ 快速开始

### 环境要求

- **Python** ≥ 3.11
- **GitHub Token**（用于调用 [GitHub Models API](https://models.github.ai/)）
- *可选*：`pdflatex`（用于 PDF 编译）

### 安装

```bash
# 克隆仓库
git clone https://github.com/SAKANA12138/research-factory.git
cd research-factory

# 安装依赖
pip install -e .

# 设置 API Key
export GITHUB_TOKEN="ghp_你的GitHub个人访问令牌"
```

### 一键运行

```bash
python -m rf.main
```

> 🎉 首次运行会自动在 `config/settings.yaml` 生成默认配置文件。

### 快速脚本

```bash
chmod +x run.sh
./run.sh --verbose
```

---

## 🔧 配置说明

编辑 `config/settings.yaml`：

```yaml
# ═══════════════════════════════════════════
# LLM 配置
# ═══════════════════════════════════════════
llm:
  api_base: "https://models.github.ai/inference"
  api_key: "${GITHUB_TOKEN}"          # 自动从环境变量读取
  default_model: "openai/gpt-4.1"    # 可选模型见下表
  temperature: 0.4
  max_tokens: 8192

# ═══════════════════════════════════════════
# 研究管线配置
# ═══════════════════════════════════════════
pipeline:
  topic: "Grafting Linear Attention into Transformer LLMs"
  target_model_size: "8B-14B"
  quantization: "INT8"
  hardware_budget_gb: 24              # 单卡显存预算

  # 阶段开关（可单独关闭某个阶段）
  phases:
    intelligence: true
    dissection: true
    debate: true
    quality: true

# ═══════════════════════════════════════════
# 数据源配置
# ═══════════════════════════════════════════
search:
  arxiv_max_results: 50
  github_stars_threshold: 100
  openreview_venues:
    - "ICLR.cc/2026/Conference"
    - "NeurIPS.cc/2025/Conference"

# ═══════════════════════════════════════════
# 输出配置
# ═══════════════════════════════════════════
output:
  output_dir: "output/"
```

### 支持的模型

| 模型 | 配置值 | 特点 |
|------|--------|------|
| GPT-4.1 | `openai/gpt-4.1` | 综合能力最强，推荐默认 |
| GPT-4.1 Mini | `openai/gpt-4.1-mini` | 更快更便宜，质量略低 |
| GPT-4.1 Nano | `openai/gpt-4.1-nano` | 最轻量，适合测试 |
| GPT-4o | `openai/gpt-4o` | ��模态支持 |
| o4-mini | `openai/o4-mini` | 推理增强型 |
| Claude Sonnet 4 | `anthropic/claude-sonnet-4` | 长文本优秀 |

---

## 🚀 使用方式

### 基本用法

```bash
# 默认配置运行
python -m rf.main

# 自定义研究主题
python -m rf.main --topic "将 Mamba SSM 模块嫁接到 LLaMA-3-8B 实现线性复杂度推理"

# 指定显存预算
python -m rf.main --vram 48

# 指定模型 + 详细日志
python -m rf.main --model openai/gpt-4.1 --verbose

# 使用自定义配置
python -m rf.main --config my_config.yaml
```

### 作为 Python 库调用

```python
import asyncio
from rf.main import run_pipeline

async def main():
    context = await run_pipeline(
        config_path="config/settings.yaml",
        overrides={
            "topic": "BitNet-style 1-bit quantization for 14B models",
            "hardware_budget_gb": 24,
        },
    )
    # 访问任意 Agent 的结构化输出
    print(context["trend_report"]["recommended_focus"])
    print(context["novelty_audit"]["novelty_score"])
    print(context["final_plan"]["go_no_go_decision"])

asyncio.run(main())
```

### 仅运行特定阶段

在 `config/settings.yaml` 中关闭不需要的阶段：

```yaml
pipeline:
  phases:
    intelligence: true
    dissection: true
    debate: false     # 跳过辩论
    quality: false    # 跳过质量检查
```

---

## 📂 项目结构

```
research_factory/
│
├── config/
│   └── settings.yaml              # 全局配置文件
│
├── rf/
│   ├── __init__.py                # 版本信息
│   ├── __main__.py                # python -m rf 入口
│   ├── main.py                    # 🎯 Pipeline 编排引擎
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py              # 🔌 统一 LLM 客户端 (GitHub Models API)
│   │
│   ├── base/
│   │   ├── __init__.py
│   │   └── agent.py               # 🧬 Agent 基类 + JSON 提取器
│   │
│   ├── phase1_intelligence/       # ──── Phase 1: 全网情报 ────
│   │   ├── __init__.py
│   │   ├── trend_analyst.py       #   🔍 热度分析师
│   │   ├── hardware_guardian.py   #   🖥️ 可行性审计师
│   │   └── novelty_auditor.py     #   🔎 查重比对官
│   │
│   ├── phase2_dissection/         # ──── Phase 2: 深度解剖 ────
│   │   ├── __init__.py
│   │   ├── source_archaeologist.py#   🏛️ 源码考古学家
│   │   ├── math_translator.py     #   📐 数学翻译官
│   │   └── conflict_detector.py   #   ⚡ 冲突检测员
│   │
│   ├── phase3_debate/             # ──── Phase 3: 对抗辩论 ────
│   │   ├── __init__.py
│   │   ├── proposer.py            #   🚀 激进派
│   │   ├── critic.py              #   👿 保守派
│   │   └── mediator.py            #   🤝 调解员
│   │
│   ├── phase4_quality/            # ──── Phase 4: 质量投稿 ────
│   │   ├── __init__.py
│   │   ├── quality_inspector.py   #   ✅ 质量自检员
│   │   ├── publication_strategist.py # 🎯 投稿战略官
│   │   └── editor_in_chief.py     #   📄 总编辑
│   │
│   └── tools/                     # ──── 外部工具集 ────
│       ├── __init__.py
│       ├── arxiv_crawler.py       #   📚 arXiv API 封装
│       ├── github_stars.py        #   ⭐ GitHub 仓库搜索
│       ├── openreview_client.py   #   📝 OpenReview API v2
│       └── latex_compiler.py      #   📄 LaTeX → PDF 编译
│
├── tests/
│   ├── test_base_agent.py         # Agent JSON 提取测试
│   ├── test_tools.py              # 工具模块测试
│   └── test_llm_config.py         # 配置加载测试
│
├── output/                        # 🎁 生成产物目录
├── pyproject.toml                 # 项目元数据 & 依赖
├── run.sh                         # 一键启动脚本
└── README.md                      # 📖 本文件
```

---

## 📊 输出产物

运行完成后，`output/` 目录包含：

| 文件 | 说明 |
|------|------|
| `rf_full_context.json` | 📦 完整管线上下文，包含全部 12 个 Agent 的结构化输出 |
| `paper.tex` | 📄 生成的 LaTeX 论文源码 |
| `paper.bib` | 📚 BibTeX 参考文献 |
| `paper.pdf` | 📕 编译后的 PDF（需安装 `pdflatex`） |
| `rf_debug.log` | 🔍 详细运行日志 |

### JSON 上下文结构

```jsonc
{
  "topic": "Grafting Linear Attention...",
  "trend_report": {                    // Phase 1: TrendAnalyst
    "trending_topics": [...],
    "recommended_focus": "Gated Linear Attention (GLA)",
    "reasoning": "..."
  },
  "hardware_audit": {                  // Phase 1: HardwareGuardian
    "vram_breakdown": { "total_gb": 18.2 },
    "fits_budget": true,
    "risk_level": "LOW"
  },
  "novelty_audit": {                   // Phase 1: NoveltyAuditor
    "novelty_score": 72,
    "novelty_level": "MEDIUM"
  },
  "source_analysis": { ... },          // Phase 2: SourceArchaeologist
  "math_translation": { ... },         // Phase 2: MathTranslator
  "conflict_report": { ... },          // Phase 2: ConflictDetector
  "proposal": { ... },                 // Phase 3: Proposer
  "criticism": {                       // Phase 3: Critic
    "overall_verdict": "BORDERLINE"
  },
  "final_plan": {                      // Phase 3: Mediator
    "go_no_go_decision": "CONDITIONAL_GO"
  },
  "quality_report": {                  // Phase 4: QualityInspector
    "total_score": 76,
    "grade": "B"
  },
  "publication_strategy": { ... },     // Phase 4: PublicationStrategist
  "paper_draft": {                     // Phase 4: EditorInChief
    "latex_source": "\\documentclass{...}",
    "bibtex_entries": "@article{...}"
  }
}
```

---

## 🔬 四阶段流水线详解

### Phase 1 — 全网情报感知 🔍

```
                    ┌──────────────┐
                    │  arXiv API   │──┐
                    └──────────────┘  │
                    ┌──────────────┐  ├──→ TrendAnalyst ──→ trend_report
                    │  GitHub API  │──┤
                    └──────────────┘  │
                    ┌──────────────┐  │
                    │  OpenReview  │──┘
                    └──────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     HardwareGuardian              NoveltyAuditor
    (VRAM 分项估算)              (近2年论文比对)
              │                           │
              ▼                           ▼
      hardware_audit                novelty_audit
```

**TrendAnalyst** 并发调用三个外部 API，收集最新论文、高 Star 仓库和会议提交数据后，交给 LLM 进行多维度热度分析。

### Phase 2 — 深度解剖建模 🔬

以流水线顺序执行：

1. **Source Archaeologist** 分析目标模型（如 LLaMA-3-8B）的 `modeling_*.py`，提取张量流向图和可嫁接模块
2. **Math Translator** 针对最优嫁接点，将替换算子的数学公式翻译为带 shape 对齐证明的 PyTorch 代码
3. **Conflict Detector** 检测 Dim / Norm / PE 三类排异风险，输出集成清单

### Phase 3 — 对抗博弈 ⚔️

模拟真实学术 Peer Review 流程：

```
Proposer（激进派）          Critic（保守派）
    │ 完整实验方案               │ 严苛审稿意见
    │ + 代码 + 预测              │ + 逐项攻击
    └──────────┬─────────────────┘
               ▼
         Mediator（调解员）
               │
               ▼
    最终实验方案 + GO/NO-GO 决策
```

### Phase 4 — 质量自检与投稿 📝

```
QualityInspector ──→ 总分 (0-100) + 各维度得分 + 问题清单
        │
        ▼
PublicationStrategist ──→ E = P(accept) × Impact 收益矩阵
        │                      投稿组合建议 (保底 + 冲击)
        ▼
EditorInChief ──→ 完整 LaTeX 源码 → pdflatex → PDF
```

---

## 🛠️ 开发与测试

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 仅运行 JSON 提取测试
pytest tests/test_base_agent.py -v

# 带覆盖率
pytest tests/ --cov=rf --cov-report=term-missing
```

### 代码风格检查

```bash
ruff check rf/ tests/
ruff format rf/ tests/
```

### 新增 Agent 指南

1. 在对应阶段目录下创建新文件，如 `rf/phase1_intelligence/my_agent.py`
2. 继承 `BaseAgent`，设置 `role` 和 `system_prompt`
3. 实现 `async def run(self, context) -> context`
4. 在 `__init__.py` 中导出
5. 在 `rf/main.py` 的 `PHASES` 字典中注册

```python
from rf.base.agent import BaseAgent

class MyNewAgent(BaseAgent):
    role = "MyNewAgent"
    system_prompt = "You are ..."

    async def run(self, context):
        result = await self._ask_structured("Analyze: ...")
        context["my_output"] = result
        return context
```

---

## ❓ 常见问题

<details>
<summary><strong>Q: 没有 GITHUB_TOKEN 怎么办？</strong></summary>

前往 [GitHub Settings → Personal Access Tokens](https://github.com/settings/tokens) 创建一个 Token，然后：

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

需要拥有 GitHub Copilot 订阅或 GitHub Models 访问权限。

</details>

<details>
<summary><strong>Q: 可以用 OpenAI / Anthropic 的原生 API 吗？</strong></summary>

可以！只需修改配置：

```yaml
llm:
  api_base: "https://api.openai.com/v1"        # 或 Anthropic 兼容端点
  api_key: "${OPENAI_API_KEY}"
  default_model: "gpt-4-turbo"
```

任何兼容 OpenAI Chat Completions API 的端点均可。

</details>

<details>
<summary><strong>Q: 运行一次大约花费多少？</strong></summary>

完整 4 阶段 12 Agent 运行约消耗 **50K-80K tokens**（视模型和主题而定）。以 GPT-4.1 为例约 $0.30-0.50 USD。使用 `gpt-4.1-nano` 可降低到 $0.05 以下。

</details>

<details>
<summary><strong>Q: PDF 编译失败怎么办？</strong></summary>

PDF 编译需要系统安装 `pdflatex`：

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# 或仅使用 .tex 源码上传到 Overleaf
```

即使 PDF 编译失败，`.tex` 源码仍会正常生成。

</details>

<details>
<summary><strong>Q: 单阶段失败会影响后续阶段吗？</strong></summary>

不会导致程序崩溃。每个 Agent 的异常会被捕获并记录到 context 中（如 `TrendAnalyst_error`），后续 Agent 会收到空的上游数据并尽力继续。

</details>

---

## 🗺️ 路线图

- [x] 4 阶段 12 Agent 基础管线
- [x] GitHub Models API 统一客户端
- [x] arXiv / GitHub / OpenReview 数据采集
- [x] 对抗博弈辩论机制
- [x] LaTeX 自动编译
- [ ] 🔄 Web UI 仪表盘（Streamlit / Gradio）
- [ ] 🔄 Agent 记忆持久化（SQLite / Redis）
- [ ] 🔄 并行 Agent 执行（Phase 1 三个 Agent 并发）
- [ ] 🔄 Semantic Scholar API 集成
- [ ] 🔄 实际代码执行沙箱（验证生成的 PyTorch 代码）
- [ ] 🔄 GitHub Actions CI/CD 自动化运行
- [ ] 🔄 多语言论文生成（中/英/日）
- [ ] 🔄 MCP (Model Context Protocol) 工具集成

---

## 🙏 致谢

- [GitHub Models](https://models.github.ai/) — LLM 推理 API
- [arXiv API](https://arxiv.org/help/api/) — 论文数据源
- [OpenReview](https://openreview.net/) — 会议论文与评审数据
- [Rich](https://github.com/Textualize/rich) — 终端美化
- [OpenAI Python SDK](https://github.com/openai/openai-python) — LLM 客户端

---

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源。

```
MIT License

Copyright (c) 2026 SAKANA12138

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with 🤖 by [SAKANA12138](https://github.com/SAKANA12138)

</div>
