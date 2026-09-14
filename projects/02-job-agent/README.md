# AI 求职分析 Agent

输入匿名简历或自有简历、岗位 JD，输出可追溯的技能覆盖与缺口，不代替招聘决策、不自动投递。

`POST /jobs/match` 是确定性工作流加可选 LLM 技能抽取；`POST /jobs/agent` 是最多 4 步的真实模型工具循环。两者被明确区分。浏览器分析按钮使用快速工作流；Agent 接口可在 `/docs` 调用。

Agent 只能调用 `extract_candidate_skills`、`match_supplied_jobs` 和 `search_job_references`。模型不能替换用于评分的原始简历或绕过白名单。LLM 总结仅供参考，排序来自原文词条覆盖；无模型时明确返回 `deterministic-workflow`。

实测本地 0.5B 模型能发出技能抽取调用，但后续规划可能输出无效 JSON。缺失的必要分析由只读确定性步骤完成，单独记录在 `fallback_trace`，状态标为 `workflow_fallback`。这不是全程模型自主规划成功的证明；保留失败与修复后的原始报告。

## 算法与工具

技能别名表统一 Python、Unity、LLM、RAG、MCP 等中英文词条，匹配证据返回原文片段。显式必需技能权重 2、加分项权重 1；缺少显式要求时从 JD 推断，并标明 `requirements_inferred`。没有可识别要求时分数为 null，而不是制造 0% 或 100%。

LLM 抽取要求证据是原文逐字子串，且确实含该技能别名；格式错误或无法提取有效词条时退回规则抽取并标记原因。词条出现不等于熟练度，否定语义、年限、学历、签证和岗位时效未进入评分。

`/jobs/search` 需要自备 Brave Search Key；返回结果只是搜索参考，不被称为已核验空缺，不自动导入评分。可将核验后的岗位通过 `/jobs/import` 保存到 SQLite。

## MCP

```bash
pip install -e '.[mcp]'
python -m ai_xr.mcp_server
```

使用官方 MCP Python SDK 的 stdio 服务，提供知识检索、技能提取和已保存岗位匹配三个只读工具。宿主配置命令为虚拟环境 Python 的绝对路径，参数为 `-m ai_xr.mcp_server`，工作目录设为仓库。不要把 stdio 输出重定向到需要 JSON-RPC 通信的 stdout。

## 验证

`tests/test_jobs.py` 覆盖匹配与证据；`tests/test_job_agent.py` 验证工具拒绝与步数限制；`tests/test_mcp.py` 验证工具注册。真实模型结果见 `evidence/integration-models.json`，测试替身只出现在单元测试，不作为真实模型成功证据。
