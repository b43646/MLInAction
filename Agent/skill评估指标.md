有，而且现在已经开始出现比较完整的 **Agent Skill Evaluation** 方法论和工具了。

如果你准备把 Skill 当成真正的工程资产，我建议不要只评估“`SKILL.md` 写得好不好”，而要评估：

> **Skill 是否让 Agent 在真实任务上表现得更好、更稳定、更省 Token，而且不会产生副作用。**

Anthropic 在 2026 年已经专门增强了 `skill-creator`，支持编写 eval、运行 benchmark、比较修改前后效果以及检测回归。([Claude][1]) OpenAI/Codex 体系目前也有 `skill-creator` 和 `plugin-eval` 相关工具，但两家的成熟度和实现方式有所不同。([GitHub][2])

---

# 一、我建议建立 6 大评估维度

可以把一个 Skill 的质量抽象成：

```text
Skill Quality
│
├── 1. Trigger Quality
├── 2. Task Success
├── 3. Instruction Following
├── 4. Output Quality
├── 5. Efficiency
└── 6. Robustness
```

其中最重要的不是“文档结构是否漂亮”，而是：

> **有 Skill 和没有 Skill，相同 Agent 在相同任务上的成功率到底提高了多少。**

---

# 二、指标 1：Trigger Accuracy

这是 Skill 特有的指标。

因为 Skill 首先要解决：

> **该不该触发？**

例如：

```text
Skill:
mysql-performance
```

准备一组测试 Prompt：

### 应该触发

```text
MySQL CPU 持续 100%，帮我分析原因。
```

```text
MySQL connection exhausted，帮我排查。
```

```text
MySQL 慢查询突然增加。
```

### 不应该触发

```text
帮我设计 MySQL 表结构。
```

```text
帮我写 MySQL CRUD。
```

```text
帮我安装 MySQL。
```

然后统计：

```text
True Positive
False Positive
True Negative
False Negative
```

最终得到：

```text
Precision
Recall
F1
```

例如：

```text
Trigger Precision = 95%
Trigger Recall    = 92%
F1                = 93.5%
```

这个指标非常重要，因为一个 Skill 如果**乱触发**，Skill 越多反而越糟糕。

OpenAI 当前的 Skill Creator 也特别强调 `description` 是 Skill 的主要触发机制，因此 description 的设计本身就是评估重点。([GitHub][2])

---

# 三、指标 2：Task Success Rate

这是最核心的。

例如你有一个：

```text
mysql-performance
```

准备 50 个真实问题：

```text
Case 01：CPU 高
Case 02：IO 高
Case 03：连接耗尽
Case 04：锁竞争
Case 05：慢查询
...
Case 50：复杂综合故障
```

定义：

```text
Success = Agent 正确找到根因并完成要求
Failure = 没有找到 / 错误结论 / 未完成
```

计算：

```text
Task Success Rate
= 成功任务数 / 总任务数
```

比如：

```text
Without Skill
36 / 50 = 72%

With Skill
45 / 50 = 90%
```

那么：

```text
Skill Lift = +18 percentage points
```

这个数字比：

> “我们的 SKILL.md 写得很专业”

有意义得多。

---

# 四、指标 3：Instruction Following

有些 Agent 最终答案可能“看起来对”，但实际上没有遵循 Skill。

比如 Skill 要求：

```text
1. 先检查 metrics
2. 再检查 logs
3. 最后提出 root cause
```

Agent 最后直接：

```text
Root cause 是连接池配置问题。
```

结论可能碰巧是对的。

但：

```text
Instruction Following = Fail
```

因此需要单独评估：

| 指标            | 是否执行 |
| ------------- | ---- |
| 按指定 workflow  | ✅    |
| 收集 evidence   | ✅    |
| 进行 hypothesis | ❌    |
| 执行 validation | ❌    |
| 输出规定格式        | ✅    |

最终：

```text
Instruction Compliance = 80%
```

---

# 五、指标 4：Output Quality

这个就是结果质量。

可以采用：

```text
Code-based grader
+
LLM-as-a-Judge
+
Human review
```

Anthropic 对 Agent Evals 的建议也是组合使用：

* code-based grader
* model-based grader
* human grader

三类评估器各有优缺点。([Anthropic][3])

---

## 例如你的“起名 Skill”

不能只判断：

```text
是否返回了 10 个名字
```

而应该：

```text
文化底蕴       0-10
音律           0-10
自然度         0-10
现代审美       0-10
独特性         0-10
负面联想       0-10
八字契合       0-10
解释质量       0-10
```

然后：

```text
Overall Score
```

甚至建立 Human Gold Set。

这和你之前做起名引擎时使用 Pearson correlation 的思路非常类似：

```text
Human Gold
     ↓
Judge
     ↓
Correlation
```

这其实就是一种非常成熟的 Skill Eval 思路。

---

# 六、指标 5：Efficiency

这一项非常容易被忽略。

一个 Skill 如果：

```text
成功率：90%
Token：100K
```

另一个：

```text
成功率：88%
Token：20K
```

未必第一个更优秀。

所以建议同时记录：

```text
Input Tokens
Output Tokens
Total Tokens
Tool Calls
Execution Time
Cost
```

然后计算：

```text
Success / 1K Tokens
```

或者：

```text
Cost per Successful Task
```

例如：

|            | No Skill | Skill |
| ---------- | -------: | ----: |
| Success    |      72% |   90% |
| Tokens     |      30K |   24K |
| Tool Calls |       14 |     9 |
| Cost       |    $0.30 | $0.24 |

这个 Skill 就非常优秀：

```text
成功率 ↑
Token ↓
Tool Calls ↓
Cost ↓
```

OpenAI 的 `plugin-eval` 已经提供了专门的 budget / token usage 分析和 benchmark 流程。([GitHub][4])

---

# 七、指标 6：Robustness / Variance

这个对于 Skill 尤其重要。

因为 Agent 有随机性。

你不能：

```text
测试一次
→ 成功
→ Skill 很好
```

而应该：

```text
同一个 Case
   ↓
Run 1
Run 2
Run 3
Run 4
Run 5
```

例如：

```text
Case #17

Run 1   PASS
Run 2   PASS
Run 3   FAIL
Run 4   PASS
Run 5   PASS
```

那么：

```text
Success Rate = 80%
```

还要关注：

```text
Variance
```

理想 Skill：

```text
PASS
PASS
PASS
PASS
PASS
```

而不是：

```text
PASS
FAIL
PASS
FAIL
PASS
```

Anthropic 新版 Skill Creator 特别强调 benchmark 和 variance analysis，就是为了发现这种问题。([GitHub][5])

---

# 八、最终可以形成一个 Skill Scorecard

我推荐你以后统一用这种：

| Dimension   | Metric                | Weight |
| ----------- | --------------------- | -----: |
| Trigger     | Precision / Recall    |    10% |
| Task        | Success Rate          |    30% |
| Instruction | Compliance            |    15% |
| Output      | Quality Score         |    20% |
| Efficiency  | Token / Cost          |    10% |
| Robustness  | Variance / Regression |    15% |

最后：

```text
Skill Score
= Σ(weight × normalized_metric)
```

例如：

```text
Trigger        93
Task Success   90
Instruction    94
Output         88
Efficiency     85
Robustness     91

Final = 90.5
```

---

# 九、但是有一个更重要的指标：Skill Lift

这个我强烈建议你加入。

不要只测：

```text
Skill = 90分
```

而应该测：

```text
Agent without Skill
        VS
Agent with Skill
```

例如：

| 指标          | Without | With |  Lift |
| ----------- | ------: | ---: | ----: |
| Success     |     72% |  90% | +18pp |
| Instruction |     61% |  94% | +33pp |
| Output      |      76 |   88 |   +12 |
| Tokens      |     31K |  24K |  -23% |
| Tool Calls  |      13 |    9 |  -31% |

这才真正证明：

> **这个 Skill 有价值。**

否则可能出现一个很尴尬的问题：

```text
Agent 本来就能做到 90%
        ↓
加了 Skill
        ↓
还是 90%
```

那么这个 Skill 其实是：

> **Zero Lift Skill**

甚至可能是负价值。

---

# 十、再进一步：Regression Test

Skill 一旦上线，就不能只测一次。

比如：

```text
Skill V1.0
Success = 90%

修改 SKILL.md

Skill V1.1
Success = 92%
```

看起来很好。

但是：

```text
Case 17：90 → 90
Case 22：95 → 60
Case 31：88 → 87
```

说明出现 Regression。

所以应该建立：

```text
golden_tasks.json
```

例如：

```json
{
  "cases": [
    {
      "id": "mysql_connection_001",
      "prompt": "MySQL connections exhausted...",
      "expected": {
        "root_cause_category": "connection_pool"
      }
    }
  ]
}
```

每次修改 Skill：

```text
git diff
   ↓
run eval
   ↓
compare
   ↓
regression detection
```

这样 Skill 就真正进入了软件工程的 CI/CD 模式。

---

# 十一、现在有哪些工具？

这里已经有比较明确的选择。

## ① Anthropic Skill Creator

如果你使用 Claude Code，这个目前是我认为**最值得研究的 Skill Eval 方案之一**。

它现在已经支持：

```text
Skill
 ↓
Test prompts
 ↓
Run
 ↓
Qualitative evaluation
 ↓
Quantitative eval
 ↓
Benchmark
 ↓
Variance analysis
 ↓
Regression detection
```

Anthropic 2026 年 3 月公开说明已经把“test → measure → refine”整合进 Skill Creator。([Claude][1])

[Anthropic Skill Creator](https://github.com/anthropics/skills/blob/main/skill-creator/SKILL.md?utm_source=chatgpt.com)

---

# 十二、② OpenAI / Codex Plugin Eval

如果你现在主要使用 Codex，那么值得关注：

```text
plugin-eval
```

它已经提供：

```text
analyze
explain-budget
init-benchmark
benchmark
measurement-plan
```

例如：

```bash
plugin-eval analyze ./skills/mysql-performance
```

或者：

```bash
plugin-eval init-benchmark ./skills/mysql-performance
```

然后：

```bash
plugin-eval benchmark ./skills/mysql-performance
```

目前 OpenAI 的 `plugin-eval` 明确支持 Skill 分析、benchmark、真实 token usage 测量以及 measurement plan。([GitHub][4])

[OpenAI plugin-eval Skill](https://github.com/openai/plugins/blob/main/plugins/plugin-eval/skills/plugin-eval/SKILL.md?utm_source=chatgpt.com)

---

# 十三、③ OpenAI Skill Creator 自带的 Validate

这个属于最低层级的：

```text
Static Validation
```

例如：

```bash
quick_validate.py
```

主要检查：

```text
YAML frontmatter
name
description
命名规范
目录结构
```

OpenAI 官方 Skill Creator 就把 `quick_validate.py` 作为 Skill 创建流程中的验证步骤。([GitHub][2])

但注意：

> **Validation ≠ Evaluation**

它只能告诉你：

```text
Skill 写得合法
```

不能告诉你：

```text
Skill 有用
```

这是两个完全不同的概念。

---

# 十四、所以我建议你建立 4 层 Eval

如果你准备把 Skill 做成真正的工程体系，我会采用：

```text
             Skill Evaluation
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
   Static       Behavioral     Quality
   Validation     Eval           Eval
       │            │            │
       ▼            ▼            ▼
  YAML/Link      Task           LLM Judge
  Structure      Success        Human
                             
                    │
                    ▼
                 Benchmark
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Accuracy    Cost      Variance
```

具体：

### Level 1 — Static

```text
SKILL.md
↓
格式
命名
引用
链接
目录
```

### Level 2 — Behavioral

```text
Prompt
↓
Agent + Skill
↓
是否完成任务
```

### Level 3 — Quality

```text
输出
↓
Code grader
LLM Judge
Human Judge
```

### Level 4 — Benchmark

```text
V1
VS
V2
VS
Baseline
```

---

# 十五、而且你现在做的项目非常适合这么搞

尤其是你前面在做的**起名引擎**。

我甚至建议你不要只做：

```text
naming-skill/
└── SKILL.md
```

而是直接：

```text
naming-skill/
│
├── SKILL.md
│
├── references/
│   ├── naming-principles.md
│   ├── aesthetic-factors.md
│   └── bazi-rules.md
│
├── scripts/
│   ├── validate_name.py
│   └── calculate_metrics.py
│
└── evals/
    ├── cases.json
    ├── graders.py
    └── benchmark.py
```

然后建立：

```text
Baseline
   ↓
Agent
   ↓
Skill V1
   ↓
Eval
   ↓
Skill V2
   ↓
Eval
   ↓
Regression
```

最终你甚至可以得到：

```text
              Naming Skill
                   │
          ┌────────┴────────┐
          │                 │
       Quality           Efficiency
          │                 │
   ┌──────┼──────┐       Token ↓
   │      │      │       Cost ↓
Culture  Trend  BaZi
   │      │      │
   └──────┼──────┘
          ↓
      Human Gold
          ↓
      Correlation
```

这就不再是“写一个 Skill”，而是在做一个真正的 **Skill Evaluation Framework**。

而且这套体系和你之前给起名引擎做的 **Human Gold Set + Pearson correlation + LLM Judge** 的思路是高度一致的——本质上都是把主观质量转化成可重复 benchmark。 ([Anthropic][3])

**如果让我给一个最简洁的结论：**

> **Skill 的静态检查只能证明“写对了”；Task Eval 才能证明“能用”；Baseline 对比才能证明“有价值”；Regression Benchmark 才能证明“长期可靠”。**

这四层最好全部建立起来。

[1]: https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills?query=workflow&utm_source=chatgpt.com "Improving skill-creator: Test, measure, and refine Agent Skills | Claude by Anthropic"
[2]: https://github.com/openai/skills/blob/main/skills/.system/skill-creator/SKILL.md?utm_source=chatgpt.com "skills/skills/.system/skill-creator/SKILL.md at main · openai/skills · GitHub"
[3]: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents \ Anthropic"
[4]: https://github.com/openai/plugins/blob/main/plugins/plugin-eval/skills/evaluate-skill/SKILL.md?utm_source=chatgpt.com "plugins/plugins/plugin-eval/skills/evaluate-skill/SKILL.md at main · openai/plugins · GitHub"
[5]: https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md?utm_source=chatgpt.com "skills/skills/skill-creator/SKILL.md at main · anthropics/skills · GitHub"
