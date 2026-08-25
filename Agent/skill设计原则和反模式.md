可以。这里我建议把 `SKILL.md` 理解成一个**给 Agent 使用的“领域操作手册（playbook）”**，而不是传统意义上的“Prompt”。

现在 Claude Code / Codex / Agent Skills 的设计方向其实高度一致：Skill 的价值是把**可重复、可验证、具有专业方法论的工作流**封装起来，并通过 metadata 按需触发，而不是把所有知识塞进一个巨大 Prompt。([Claude Platform][1])

---

# 一、先建立一个正确认知：SKILL.md 到底是什么？

我会把它定义成：

> **SKILL.md = 任务边界 + 决策规则 + 工作流 + 质量标准 + 必要示例**

而不是：

> “告诉 AI 这个领域有哪些知识。”

例如：

**不好的 Skill：**

```text
# Kubernetes Skill

Kubernetes 是一个容器编排平台。

它包含：
- Pod
- Deployment
- Service
- ConfigMap
- Secret
- Ingress
...
```

这其实更像一篇 Kubernetes 教程。

**好的 Skill：**

```text
# Kubernetes Troubleshooting

当用户要求排查 Kubernetes Pod 启动失败、CrashLoopBackOff、
ImagePullBackOff 或 Service 不通时使用。

## Workflow

1. 先确认 namespace 和 workload。
2. 检查 Pod 状态。
3. 如果 Pod 未启动：
   - 查看 events
   - 查看 container status
   - 查看 image
4. 如果 Pod 已启动但服务不可用：
   - 检查 Service selector
   - 检查 Endpoints
   - 检查 NetworkPolicy
5. 不要在证据不足时直接修改资源。

## Validation

修复后必须：
- 检查 Pod Ready
- 检查 Service Endpoints
- 执行一次实际请求验证

## Escalation

如果无法确定根因：
- 不要猜测
- 输出已确认事实
- 输出需要进一步检查的信息
```

后者才真正是在**教 Agent 怎么工作**。

---

# 二、最核心的 7 个设计原则

## 原则 1：一个 Skill 只解决一个“可复用任务”

这是最重要的一条。

官方建议也是让 Skill 保持 focused，一个 Skill 聚焦一个工作。([Claude][2])

### ❌ 反模式：万能 Skill

```text
# enterprise-ai-engineer

这个 Skill 帮助 Agent：

- 写代码
- 做架构设计
- 部署 Kubernetes
- 分析数据库
- 写 Terraform
- 写 PPT
- 做安全审计
- 写测试
- 做 DevOps
```

问题是：

**Skill 的边界完全消失。**

最后 Agent 不知道：

> “什么时候应该使用这个 Skill？”

---

### ✅ 推荐

拆成：

```text
skills/
├── kubernetes-troubleshooting/
│   └── SKILL.md
│
├── terraform-oci/
│   └── SKILL.md
│
├── mysql-performance/
│   └── SKILL.md
│
├── architecture-review/
│   └── SKILL.md
│
└── incident-report/
    └── SKILL.md
```

这样每个 Skill 都有非常清晰的：

> **Trigger → Workflow → Output**

---

# 三、原则 2：Description 比你想象的重要得多

Skill 通常存在一个“渐进式加载”机制：

```text
Level 1
Metadata
  ↓
name + description

Level 2
SKILL.md
  ↓
Instructions

Level 3
references / scripts / assets
  ↓
按需加载
```

也就是说，Agent 一开始并不会把所有 Skill 内容都塞进上下文，而是首先依赖 metadata 判断：

> “这个 Skill 是否与当前任务相关？”

官方文档明确强调 description 同时应该说明：

**做什么 + 什么时候使用。** ([Claude Platform][1])

---

## ❌ 差的 Description

```yaml
---
name: mysql
description: MySQL database skill
---
```

问题：

Agent 不知道什么时候应该触发。

---

## ❌ 还是不够好

```yaml
---
name: mysql-performance
description: Help with MySQL performance.
---
```

仍然太宽。

---

## ✅ 更好的写法

```yaml
---
name: mysql-performance
description: Diagnose MySQL performance problems including slow queries, high CPU,
lock contention, connection exhaustion, and replication lag. Use when the user
asks to investigate, troubleshoot, or optimize an existing MySQL workload.
---
```

这里已经出现了：

```text
What
↓
MySQL performance problems

具体范围
↓
slow query
CPU
lock
connection
replication

Trigger
↓
investigate / troubleshoot / optimize
```

这就是一个好的 Skill Trigger。

---

# 四、原则 3：SKILL.md 应该写“怎么做”，而不是写“知识百科”

这是很多人最容易犯的错误。

假设你做：

```text
oci-architecture-review
```

### ❌ 不要这样写

```text
OCI 是 Oracle Cloud Infrastructure。

OCI 包含：

VCN
Compute
Load Balancer
Object Storage
OKE
Database
...
```

这些都是**知识**。

LLM 本身通常已经知道很多。

Skill 真正应该补充的是：

```text
什么时候选择 NLB？
什么时候选择 LB？

什么时候使用 Private Subnet？

如何判断是否需要 NAT Gateway？

如何检查 HA？

如何检查 SPOF？

架构评审必须输出什么？
```

也就是：

> **Skill 应该补充模型的“工作方法”，而不是重复模型已有的“百科知识”。**

Codex 的 Skill Creator 也明确强调：不要重复模型已经知道的信息，只增加模型真正需要的上下文。([GitHub][3])

---

# 五、原则 4：给 Agent “决策规则”，不要只给命令

这是高级 Skill 和普通 Prompt 最大的区别之一。

例如：

### ❌ 低质量

```text
检查 MySQL：

1. SHOW PROCESSLIST;
2. SHOW ENGINE INNODB STATUS;
3. SHOW VARIABLES;
```

这是命令清单。

---

### ✅ 高质量

```text
## Connection Exhaustion

如果用户报告 connection exhausted：

1. 检查当前连接数。
2. 检查连接上限。
3. 检查 ProxySQL / connection pool。
4. 对比：
   active connections
   max_connections
   backend connections
5. 如果应用连接数明显低于 MySQL max_connections，
   不要直接提高 max_connections。
6. 优先判断连接池、代理层或连接泄漏。
```

这里 Agent 获得的是：

```text
现象
 ↓
诊断路径
 ↓
判断条件
 ↓
下一步动作
```

这才是 Skill。

---

# 六、原则 5：对“脆弱操作”降低自由度

这个原则非常重要。

Codex 官方 Skill Creator 对这一点总结得很好：

> 任务越容易出错、越需要一致性，就应该给 Agent 越低的自由度。([GitHub][3])

可以把 Agent 的自由度理解成：

```text
高自由度
    ↓
中自由度
    ↓
低自由度
```

---

## 高自由度任务

例如：

```text
分析这个系统的架构问题。

可以根据实际情况选择分析方法。
```

OK。

因为架构分析本身允许多种方法。

---

## 中自由度

例如：

```text
设计 Kubernetes Deployment。

优先使用：
Deployment
RollingUpdate
readinessProbe
livenessProbe

如果业务特殊，再调整。
```

---

## 低自由度

例如：

```text
生产数据库迁移。

必须：

1. 检查 backup
2. 检查 replication
3. 验证目标库
4. 执行 migration
5. 验证 schema
6. 验证应用
7. 保留 rollback 方法

禁止：

- 未备份直接 migration
- 未确认生产环境直接执行
- 删除旧表作为第一步
```

这种任务千万不要写：

```text
“你可以根据情况自由处理。”
```

因为 Agent 的自由度太高。

---

# 七、原则 6：一定要写“为什么”

这是非常容易被忽略的一点。

例如：

### ❌

```text
不要直接修改 max_connections。
```

Agent 可能会问：

> 为什么？

---

### ✅

```text
不要在 connection exhausted 的情况下直接提高
max_connections。

原因：

连接耗尽可能发生在 ProxySQL、应用连接池或连接泄漏层。
单纯提高 MySQL max_connections 可能只是延迟问题，
同时可能增加数据库内存压力。
```

这样 Agent 遇到新的情况时，可以进行**规则泛化**。

Anthropic 的 Skill Specification 也明确建议：重要指令不仅说明 What，还应该说明 Why。([Mintlify][4])

---

# 八、原则 7：把 SKILL.md 当成“代码”一样测试

这个原则我尤其建议你重视。

不要：

```text
写完 SKILL.md
↓
觉得写得挺好
↓
结束
```

而应该：

```text
SKILL.md
 ↓
Trigger Test
 ↓
Workflow Test
 ↓
Edge Case Test
 ↓
Output Test
 ↓
Regression Test
```

例如你写：

```text
mysql-performance
```

至少准备：

```text
Case 1
MySQL CPU 100%

Case 2
Connection exhausted

Case 3
Slow query

Case 4
Lock contention

Case 5
Replication lag

Case 6
其实不是 MySQL 的问题
```

然后观察 Agent：

```text
Skill 是否正确触发？
        ↓
诊断顺序是否正确？
        ↓
有没有跳步骤？
        ↓
有没有瞎猜？
        ↓
有没有执行危险操作？
        ↓
最终输出是否符合要求？
```

官方也建议使用真实任务测试 Skill 的触发和行为。([GitHub][3])

---

# 九、SKILL.md 最推荐的结构

我比较推荐你采用这个结构：

```text
SKILL.md

1. Frontmatter
2. Purpose
3. When to use
4. When NOT to use
5. Inputs
6. Workflow
7. Decision rules
8. Constraints / Safety
9. Validation
10. Output
11. Examples
12. References
```

例如：

```markdown
---
name: mysql-performance
description: Diagnose MySQL performance problems including slow queries,
high CPU, lock contention, connection exhaustion, and replication lag.
Use when investigating or optimizing an existing MySQL workload.
---

# MySQL Performance Diagnosis

## Purpose

Provide a systematic method for diagnosing MySQL performance issues
without making unsupported assumptions.

## When to use

Use when:

- MySQL CPU is high
- queries are slow
- connections are exhausted
- lock contention occurs
- replication lag increases

## When NOT to use

Do not use for:

- schema design
- application-level debugging
- database migration
- backup / restore

## Inputs

Collect:

- MySQL version
- workload symptoms
- current metrics
- relevant configuration
- recent changes

## Workflow

### Step 1: Establish the symptom

Determine whether the issue is:

- CPU
- IO
- memory
- connection
- lock
- query
- replication

Do not change configuration before identifying the symptom class.

### Step 2: Collect evidence

...

### Step 3: Diagnose

...

## Decision Rules

If:

CPU high + query latency high
→ inspect expensive queries first.

If:

connections high + backend connections low
→ inspect connection pool / proxy.

If:

replication lag high + IO saturation
→ inspect disk throughput before changing replication settings.

## Constraints

Do not:

- increase limits without evidence
- kill production queries without confirmation
- change configuration before identifying the bottleneck

## Validation

After remediation:

1. Re-check metrics.
2. Verify symptom improvement.
3. Confirm no new errors.
4. Compare against baseline.

## Output

Return:

1. Symptoms
2. Evidence
3. Root cause
4. Remediation
5. Validation
6. Remaining risks
```

这个结构已经非常接近一个真正的 **Agent operational playbook**。

---

# 十、最重要的反模式

下面这些，我建议你以后审查 Skill 时直接当成 Checklist。

| 反模式                 | 问题          | 改进                     |
| ------------------- | ----------- | ---------------------- |
| 万能 Skill            | 边界模糊        | 一个 Skill 一个核心任务        |
| 巨型 SKILL.md         | Context 污染  | 拆 references           |
| 百科全书                | 浪费 token    | 只写模型缺失的知识              |
| 只写命令                | 没有决策能力      | 增加 Decision Rules      |
| 只写 What             | 遇到边界情况容易失效  | 增加 Why                 |
| 没有 Trigger          | 容易误触发       | 明确 When to use         |
| 没有 Negative Trigger | Skill 到处触发  | 明确 When NOT to use     |
| 全部 MUST             | Agent 过度僵化  | 区分 MUST / SHOULD / MAY |
| 完全自由发挥              | 生产操作风险高     | 对脆弱操作降低自由度             |
| 没有 Validation       | Agent 自认为完成 | 增加验证步骤                 |
| 没有 Examples         | Agent 理解偏差  | 给少量高价值示例               |
| 重复 reference 内容     | Token 浪费    | Single Source of Truth |
| 写大量背景故事             | 对执行无帮助      | 删除                     |
| 没有测试                | 质量不可控       | 建立 Skill eval          |

---

# 十一、一个特别典型的反模式：把 Skill 写成“Prompt 墙”

例如：

```markdown
# My Coding Skill

你是一名资深软件工程师。

你必须：

- 写高质量代码
- 遵循 SOLID
- 遵循 DRY
- 遵循 KISS
- 注意安全
- 注意性能
- 注意可维护性
- 注意异常处理
- 注意日志
- 注意测试
- 注意代码规范
- 注意架构设计
- 注意用户体验
- ...
```

这看起来很专业。

实际上非常弱。

因为：

```text
原则很多
↓
行动规则很少
↓
没有 workflow
↓
没有 decision tree
↓
没有 validation
```

Agent 最后还是不知道：

> **“所以我现在到底应该做什么？”**

---

# 十二、反过来，一个非常好的 Skill 长什么样？

例如：

```text
用户：
“生产环境 Redis 延迟突然从 2ms 上升到 80ms。”
```

好的 Skill 应该让 Agent 自动进入：

```text
             Redis Latency
                   │
                   ▼
           Establish baseline
                   │
                   ▼
          latency / throughput
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
         CPU       IO      Network
          │        │        │
          ▼        ▼        ▼
      saturation  disk    packet
          │
          ▼
      slow commands?
          │
     ┌────┴────┐
     ▼         ▼
    YES        NO
     │          │
     ▼          ▼
 inspect      inspect
 commands     infrastructure
```

然后：

```text
Evidence
   ↓
Hypothesis
   ↓
Test hypothesis
   ↓
Root cause
   ↓
Remediation
   ↓
Validation
```

这才是真正的 Skill。

---

# 十三、Skill 的“知识”和“能力”应该怎么拆？

这是我认为你在做 Agent 项目时尤其值得采用的一种架构：

```text
                SKILL.md
                   │
        ┌──────────┼──────────┐
        │          │          │
      Rules      Workflow   Output
        │          │          │
        └──────────┼──────────┘
                   │
           ┌───────┴────────┐
           │                │
      references/        scripts/
           │                │
      ┌────┼────┐           │
      │    │    │           │
    API  Schema Policy    deterministic
    Docs         Docs      operations
```

也就是说：

### SKILL.md

负责：

> **怎么做**

### references/

负责：

> **需要查什么**

### scripts/

负责：

> **必须稳定执行什么**

### assets/

负责：

> **输出时需要什么资源**

OpenAI 的 Skill 结构也明确采用这种渐进式设计：核心操作放在 `SKILL.md`，详细参考资料放到 `references/`，确定性操作放到 `scripts/`，从而避免把所有内容一次性塞进上下文。([GitHub][5])

---

# 十四、还有一个非常关键的原则：不要让 Skill 和 Agent 本身抢职责

可以简单理解：

```text
LLM
│
├── General reasoning
├── General knowledge
└── Planning
       │
       ▼
SKILL
│
├── Domain workflow
├── Decision rules
├── Constraints
└── Quality criteria
       │
       ▼
Tools / Scripts
│
├── Deterministic execution
├── API
├── CLI
└── Validation
```

例如：

**不要在 Skill 里自己实现复杂计算逻辑：**

```text
请计算所有 CIDR 地址范围……
```

如果这是一个确定性计算，最好：

```text
SKILL
 ↓
调用 script
 ↓
script 计算
 ↓
返回结果
```

Agent Skills 官方设计本身就强调：

> 灵活判断交给 instructions，确定性操作交给 scripts。([Claude Platform][1])

---

# 十五、我给你一个“Skill 质量公式”

如果以后你让 Coding Agent 批量创建 Skill，可以直接让它按照这个标准检查：

```text
Skill Quality
=
Trigger Accuracy
×
Task Specificity
×
Workflow Clarity
×
Decision Quality
×
Validation
÷
Context Cost
```

其中最容易被忽略的是：

### Context Cost

Skill 不是越详细越好。

例如：

```text
3000 行 Skill
```

不一定比：

```text
200 行 Skill
+
3 个 references
+
2 个 scripts
```

更好。

实际上后者往往更强。

官方建议 `SKILL.md` 保持精简，在接近较大规模时把详细内容移到 reference 文件；Codex 也强调 context window 是共享资源。([Mintlify][4])

---

# 十六、结合你现在做 Coding Agent，我尤其建议这样设计

你最近实际上已经进入了一个更高级的阶段：

```text
Prompt
   ↓
Agent
   ↓
Skills
   ↓
Tools
   ↓
Knowledge
   ↓
Evaluation
```

所以以后不要再简单地想：

> “我要给 Coding Agent 写一个很好的 Prompt。”

而应该想：

> **“我要把这个任务的专家工作流固化成一个可复用 Skill。”**

例如你做你现在的**宝宝起名引擎**，就完全可以拆成：

```text
skills/
├── naming-data-cleaning/
│   └── SKILL.md
│
├── naming-aesthetic-analysis/
│   └── SKILL.md
│
├── naming-candidate-generation/
│   └── SKILL.md
│
├── naming-quality-evaluation/
│   └── SKILL.md
│
├── naming-bazi-filter/
│   └── SKILL.md
│
└── naming-engine-evaluation/
    └── SKILL.md
```

其中：

```text
naming-data-cleaning
```

不是告诉 Agent：

> “中国名字有哪些文化。”

而是告诉它：

```text
什么时候清洗
→ 清洗什么
→ canonical 怎么定义
→ 重复怎么处理
→ C 级如何处理
→ 负面谐音怎么检测
→ 数据质量如何验证
→ 输出什么统计指标
```

这就是 Skill 的正确思路。

---

## 最后浓缩成 10 条规则

如果你以后只记住这些：

1. **一个 Skill 一个核心任务。**
2. **Description = 做什么 + 什么时候触发。**
3. **SKILL.md 写工作方法，不写百科全书。**
4. **优先写 Workflow，而不是知识堆积。**
5. **给 Agent Decision Rules。**
6. **重要规则解释 Why。**
7. **危险/脆弱操作降低 Agent 自由度。**
8. **详细知识放 `references/`，确定性操作放 `scripts/`。**
9. **必须有 Validation，而不是“做完就算”。**
10. **把 Skill 当代码测试：Trigger、Workflow、Edge Case、Regression 都要测。**

我认为最核心的一句话其实是：

> **好的 SKILL.md 不是“让 AI 知道更多”，而是“让 AI 在一个特定任务上稳定地按照专家的方法做事”。**

这也是为什么 Skill 和普通 System Prompt / Rules / README 的定位完全不同。([Claude Platform][1])

[1]: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview?utm_source=chatgpt.com "Agent Skills - Claude Platform Docs"
[2]: https://claude.com/docs/skills/how-to?utm_source=chatgpt.com "Creating custom skills - Claude.ai Documentation"
[3]: https://github.com/openai/codex/blob/main/codex-rs/skills/src/assets/samples/skill-creator/SKILL.md?utm_source=chatgpt.com "codex/codex-rs/skills/src/assets/samples/skill-creator/SKILL.md at main · openai/codex · GitHub"
[4]: https://www.mintlify.com/anthropics/skills/spec/overview?utm_source=chatgpt.com "Agent Skills Specification Overview - Anthropic Skills"
[5]: https://github.com/openai/skills/blob/main/skills/.system/skill-creator/SKILL.md?utm_source=chatgpt.com "skills/skills/.system/skill-creator/SKILL.md at main · openai/skills · GitHub"
