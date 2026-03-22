<!-- language-switch:start -->
[English](./README.md) | [中文](./README.zh-CN.md)
<!-- language-switch:end -->

<div align="center" id="top">
<a href="https://agno.com">
<picture>
<source media="(首选颜色方案：深色)" srcset="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-dark.svg">
<source media="(prefers-color-scheme: light)" srcset="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg">
<img src="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg" alt="Agno">
</picture>
</a>
</div>

<p align="center">
代理软件的编程语言。<br/>
大规模构建、运行和管理多代理系统。
</p>

<div align="center">
<a href="https://docs.agno.com">Docs</a>
<span> • </span>
<a href="https://github.com/agno-agi/agno/tree/main/cookbook">Cookbook</a>
<span> • </span>
<a href="https://docs.agno.com/first-agent">Quickstart</a>
<span> • </span>
<a href="https://www.agno.com/discord">Discord</a>
</div>

## 阿格诺是什么？

软件正在从确定性请求-响应转向规划、调用工具、记住上下文和做出决策的推理系统。 Agno 是正确构建该软件的语言。它提供：

|层 |责任|
|-------|----------------|
| **SDK** |智能体、团队、工作流程、内存、知识、工具、护栏、审批流程 |
| **发动机** |模型调用、工具编排、结构化输出、运行时执行 |
| **代理操作系统** |流 API、隔离、身份验证、批准执行、跟踪、控制平面 |

## 快速入门

构建一个有状态的、使用工具的代理，并将其用作约 20 行生产 API。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.os import AgentOS
from agno.tools.mcp import MCPTools

agno_assist = Agent(
    name="Agno Assist",
    model=Claude(id="claude-sonnet-4-6"),
    db=SqliteDb(db_file="agno.db"),
    tools=[MCPTools(url="https://docs.agno.com/mcp")],
    add_history_to_context=True,
    num_history_runs=3,
    markdown=True,
)

agent_os = AgentOS(agents=[agno_assist], tracing=True)
app = agent_os.get_app()
```

运行它：

```bash
export ANTHROPIC_API_KEY="***"

uvx --python 3.12 \
  --with "agno[os]" \
  --with anthropic \
  --with mcp \
  fastapi dev agno_assist.py
```

在大约 20 行内，​​您将得到：
- 具有流响应的有状态代理
- 每用户、每会话隔离
- http://localhost:8000 的生产 API
- 本机追踪

连接到 [AgentOS UI](https://os.agno.com) 以监视、管理和测试您的智能体。

1. 打开 [os.agno.com](https://os.agno.com) 并登录。
2. 单击顶部导航中的**“添加新操作系统”**。
3. 选择 **“本地”** 连接到本地 AgentOS。
4. 输入您的端点 URL（默认值：`http://localhost:8000`）。
5. 将其命名为“本地 AgentOS”。
6. 单击**“连接”**。

https://github.com/user-attachments/assets/75258047-2471-4920-8874-30d68c492683

打开聊天，选择您的代理，然后询问：

> 阿格诺是什么？

代理从 Agno MCP 服务器检索上下文并以可靠的答案进行响应。

https://github.com/user-attachments/assets/24c28d28-1d17-492c-815d-810e992ea8d2

您可以使用完全相同的架构在生产中运行多代理系统。

## 为什么是阿格诺？

代理软件引入了三个基本转变。

### 新的交互模式

传统软件接收请求并返回响应。代理实时传输推理、工具调用和结果。他们可以在执行过程中暂停，等待批准，然后再恢复。

Agno 将流式处理和长时间运行的执行视为一流的行为。

### 新的治理模式

传统系统执行预先编写的预定义决策逻辑。代理动态选择操作。有些行为风险较低。有些需要用户批准。有些需要行政权力。

Agno 允许您定义谁决定什么作为代理定义的一部分，其中：

- 审批工作流程
- 人在回路
- 审核日志
- 运行时执行

### 新的信任模式

传统系统被设计为可预测的。每个执行路径都是预先定义的。代理将概率推理引入执行路径。

Agno 在引擎本身中建立了信任：

- Guardrails 作为执行的一部分运行
- 评估集成到代理循环中
- 跟踪和审计日志是一流的

## 专为生产而打造

Agno 在您的基础设施中运行，而不是在我们的基础设施中。

- 无状态、水平可扩展的运行时。
- 50 多个 API 和后台执行。
- 每用户和每会话隔离。
- 运行时批准强制执行。
- 本机跟踪和完整的可审计性。
- 会话、内存、知识和痕迹存储在数据库中。

您拥有该系统。您拥有数据。您定义规则。

## 您可以构建什么

Agno 为基于上述相同原语构建的真正代理系统提供支持。

- [**Pal →**](https://github.com/agno-agi/pal) 了解您偏好的个人智能体。
- [**Dash →**](https://github.com/agno-agi/dash) 一种基于六层上下文的自学习数据智能体。
- [**Scout →**](https://github.com/agno-agi/scout) 管理企业上下文知识的自学习上下文智能体。
- [**Gcode →**](https://github.com/agno-agi/gcode) 一种随着时间的推移而改进的 IDE 后编码智能体。
- [**投资团队→**](https://github.com/agno-agi/investment-team) 一个多代理投资委员会，负责辩论和分配资本。

单一智能体。协调的团队。结构化的工作流程。全部构建在一个架构上。

## 开始使用

1. [阅读文档](https://docs.agno.com)
2. [建立你的第一个代理](https://docs.agno.com/first-agent)
3. 探索[食谱](https://github.com/agno-agi/agno/tree/main/cookbook)

## IDE集成

将 Agno 文档添加为编码工具中的源：

**光标：** 设置 → 索引和文档 → 添加 `https://docs.agno.com/llms-full.txt`

还适用于 VSCode、Windsurf 和类似工具。

## 贡献

请参阅[贡献指南](https://github.com/agno-agi/agno/blob/main/CONTRIBUTING.md)。

## 遥测

Agno 记录哪些模型提供程序用于确定更新的优先级。使用 `AGNO_TELEMETRY=false` 禁用。

<p align="right"><a href="#top">↑ 返回顶部</a></p>
