有，而且如果你是在设计 **Agent Skill / `SKILL.md` 体系**，我建议不要只关注“文件怎么写”，更应该关注**Skill 内部采用什么行为设计范式**。

目前比较实用的可以归纳成下面 **8 种范式**。其中前 5 种最值得掌握。

---

# 1. Workflow Pattern：标准工作流范式

这是最基础、也是最常用的。

核心：

```text
Input
  ↓
Understand
  ↓
Analyze
  ↓
Execute
  ↓
Validate
  ↓
Output
```

例如数据库故障诊断：

```text
用户问题
  ↓
确认现象
  ↓
收集指标
  ↓
定位瓶颈
  ↓
提出假设
  ↓
验证假设
  ↓
执行修复
  ↓
验证修复
  ↓
输出 RCA
```

SKILL.md 可以设计成：

```markdown
## Workflow

### Step 1: Establish symptoms

### Step 2: Collect evidence

### Step 3: Form hypotheses

### Step 4: Validate hypotheses

### Step 5: Remediate

### Step 6: Validate remediation

### Step 7: Report
```

### 什么时候用？

几乎所有 Skill 都可以使用。

特别适合：

* Troubleshooting
* Code review
* Architecture review
* Data processing
* Security audit
* Report generation

---

# 2. Decision Tree Pattern：决策树范式

如果一个任务存在明显的：

> **“如果 A，就做 B；如果 C，就做 D”**

那么不要只写 workflow，应该使用 Decision Tree。

例如：

```text
Redis 延迟升高
      │
      ▼
   CPU 高？
   /     \
 YES      NO
  │        │
  ▼        ▼
CPU分析   IO高？
          /   \
        YES    NO
         │      │
         ▼      ▼
       IO分析  Network
```

对应 Skill：

```markdown
## Decision Rules

IF CPU > 80%
→ inspect CPU saturation

IF CPU normal AND IO latency high
→ inspect disk

IF CPU and IO normal
→ inspect network

IF all infrastructure metrics normal
→ inspect slow commands
```

### 什么时候用？

特别适合：

* 故障诊断
* 安全分析
* 数据质量检查
* 架构选型
* 技术方案决策

---

# 3. State Machine Pattern：状态机范式

这个非常适合 Agent。

很多复杂任务其实不是：

```text
Step 1 → Step 2 → Step 3
```

而是：

```text
State A
  ↓
State B
  ↓
State C
  ↓
可能失败
  ↓
Retry / Rollback / Escalate
```

例如：

```text
INIT
 │
 ▼
DISCOVERY
 │
 ▼
ANALYSIS
 │
 ├── insufficient evidence → DISCOVERY
 │
 ▼
PLAN
 │
 ├── high risk → HUMAN_APPROVAL
 │
 ▼
EXECUTION
 │
 ├── failed → ROLLBACK
 │
 ▼
VALIDATION
 │
 ├── failed → INVESTIGATION
 │
 ▼
DONE
```

这种设计比简单的 Step 1/2/3 更强。

---

### 什么时候特别有价值？

例如你做：

**生产环境运维 Agent**

```text
DISCOVER
→ DIAGNOSE
→ PLAN
→ APPROVAL
→ EXECUTE
→ VERIFY
→ CLOSE
```

因为每一步都有状态和转移条件。

---

# 4. Plan → Execute → Verify Pattern

这是我非常推荐 Coding Agent 使用的一种。

核心：

```text
Plan
 ↓
Execute
 ↓
Verify
```

而不是：

```text
Think
 ↓
Do
 ↓
Done
```

例如 Coding Agent 修 Bug：

```text
1. Understand issue
2. Inspect repository
3. Identify root cause
4. Create implementation plan
5. Modify code
6. Run tests
7. Inspect failures
8. Fix
9. Run regression tests
10. Summarize
```

对应：

```markdown
## Execution Protocol

### Plan

Before modifying code:

- identify affected modules
- identify dependencies
- identify expected behavior

### Execute

Implement the smallest change that addresses the root cause.

### Verify

Run:

1. targeted tests
2. related tests
3. full regression if appropriate

Do not claim completion until verification succeeds.
```

这个模式对于 **Coding Agent / DevOps Agent / Data Agent** 都非常重要。

---

# 5. Evidence → Hypothesis → Validation Pattern

这个范式尤其适合**诊断类 Skill**。

核心思想：

> **禁止 Agent 从现象直接跳到结论。**

例如：

```text
Evidence
   ↓
Hypothesis
   ↓
Test
   ↓
Result
   ↓
Conclusion
```

比如：

```text
现象：
MySQL connections = 4000

不能直接：
→ MySQL max_connections 太小
```

而应该：

```text
Evidence:
ProxySQL connections = 4000
MySQL active connections = 235

↓

Hypothesis:
connection pool / ProxySQL 层存在连接堆积

↓

Validation:
检查：
stats_mysql_connection_pool
mysql_servers
connection pool settings

↓

Conclusion:
问题发生在 ProxySQL，而非 MySQL backend
```

这个范式可以显著降低 Agent **“看见症状就猜根因”** 的问题。

---

# 6. Guardrail Pattern：护栏范式

这个不是 workflow，而是**限制 Agent 行为**。

例如生产数据库 Skill：

```markdown
## Safety Constraints

MUST:

- inspect current state before modification
- create backup before destructive operations
- validate target environment

MUST NOT:

- drop tables without confirmation
- modify production configuration without approval
- kill queries without identifying owner

SHOULD:

- prefer reversible changes
- minimize blast radius
```

可以理解成：

```text
          Agent
            │
     ┌──────┴──────┐
     │             │
   Allowed       Forbidden
     │             │
     ▼             ▼
  execute        blocked
```

---

### 一个重要技巧

不要把所有东西都写成：

```text
MUST
MUST
MUST
MUST
MUST
```

否则 Skill 会变成“规则泥潭”。

应该区分：

```text
MUST
SHOULD
MAY
MUST NOT
```

例如：

```text
MUST:
验证生产环境

SHOULD:
优先选择可回滚方案

MAY:
根据情况选择监控工具
```

---

# 7. Progressive Disclosure Pattern：渐进式披露

这个是 **Skill 架构层面**非常重要的范式。

不要：

```text
SKILL.md
  ↓
5000 行
  ↓
所有知识
```

而应该：

```text
SKILL.md
   │
   ├── Core workflow
   ├── Decision rules
   └── Safety
          │
          ▼
     references/
       ├── mysql.md
       ├── proxysql.md
       ├── examples.md
       └── troubleshooting.md
          │
          ▼
       scripts/
       ├── collect_metrics.py
       └── validate.py
```

也就是：

```text
核心规则
 ↓
需要时加载详细知识
 ↓
需要时调用工具
```

这可以显著降低 Context 消耗。

---

# 8. Critic / Judge Pattern：自我评审范式

这个非常适合你现在做的**起名引擎**。

流程：

```text
Generate
   ↓
Evaluate
   ↓
Critique
   ↓
Improve / Filter
   ↓
Final
```

例如：

```text
候选名字
   ↓
文化评价
   ↓
音律评价
   ↓
自然度评价
   ↓
流行度评价
   ↓
八字评价
   ↓
负面联想检测
   ↓
综合评分
```

甚至可以进一步：

```text
Generator
     ↓
Candidate Pool
     ↓
Judge
     ↓
Critic
     ↓
Re-rank
     ↓
Final
```

这种模式非常适合：

* Naming
* Content generation
* Code generation
* Architecture design
* Marketing copy
* Research

---

# 9. 实际项目里，最好不是只选一种

真正成熟的 Skill 通常是**组合范式**。

例如你的“宝宝起名 Skill”，我会设计成：

```text
                 Naming Skill
                      │
              ┌───────┴───────┐
              │               │
          Workflow        Guardrails
              │               │
              ▼               ▼
          Generate       禁止生僻字
              │          禁止负面谐音
              ▼          禁止明显歧义
          Hard Filter
              │
              ▼
       Evidence / Features
              │
              ▼
            Judge
              │
              ▼
          Multi-factor
           Evaluation
              │
              ▼
          Re-ranking
              │
              ▼
          Ba Zi Filter
              │
              ▼
           Validation
              │
              ▼
            Output
```

实际上就是：

```text
Workflow
   +
Decision Tree
   +
Guardrail
   +
Judge
   +
Validation
```

---

# 10. 我建议你建立一个“Skill Pattern Catalog”

如果你后面要让 Coding Agent 大规模创建 Skill，最好不要每次让它自由发挥。

可以先规定：

| Pattern                | 核心用途       | 典型场景            |
| ---------------------- | ---------- | --------------- |
| Workflow               | 顺序执行       | 数据处理、报告         |
| Decision Tree          | 条件决策       | 故障诊断            |
| State Machine          | 复杂状态       | 运维 Agent        |
| Plan-Execute-Verify    | 执行任务       | Coding Agent    |
| Evidence-Hypothesis    | 根因分析       | Troubleshooting |
| Guardrail              | 风险控制       | Production      |
| Progressive Disclosure | 控制 Context | 所有复杂 Skill      |
| Judge-Critic           | 质量评估       | 生成类任务           |

然后规定：

> **每个 Skill 至少选择一个主范式，可以组合 2～4 个辅助范式，但不要无脑全部使用。**

这是非常重要的。

---

## 最后给你一个判断方法

拿到一个新任务，不要马上写 SKILL.md。

先问：

```text
这个任务本质是什么？
```

如果答案是：

**“按照固定步骤完成”**

→ `Workflow`

**“根据不同情况采取不同方案”**

→ `Decision Tree`

**“任务过程中有多个状态和回退”**

→ `State Machine`

**“要修改东西，并证明修改成功”**

→ `Plan → Execute → Verify`

**“容易误判根因”**

→ `Evidence → Hypothesis → Validation`

**“可能造成生产风险”**

→ `Guardrail`

**“知识很多但不能全部放上下文”**

→ `Progressive Disclosure`

**“需要生成后再评估”**

→ `Judge / Critic`

### 我认为最值得你掌握的组合是：

```text
          Skill
            │
       ┌────┴────┐
       │ Workflow│
       └────┬────┘
            │
      Decision Rules
            │
       Guardrails
            │
      Execute / Judge
            │
        Validation
```

**Workflow 决定“怎么走”，Decision 决定“走哪条路”，Guardrail 决定“什么不能做”，Judge 决定“做得好不好”，Validation 决定“到底做没做对”。**

这五个组合起来，基本就是一个成熟 Agent Skill 的骨架。
