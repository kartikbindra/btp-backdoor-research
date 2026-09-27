# Campaign 001

This directory contains the master prompt for the first research campaign.

Recommended Antigravity workflow:

1. Open the `btp-research` repository as the workspace.
2. Ensure `.agents/agents/` contains the custom agents.
3. Ensure `AGENTS.md` is at repository root.
4. Start a new Antigravity conversation.
5. Run `/teamwork-preview` and provide the content of `MASTER_PROMPT.md` as the project brief.
6. Review the generated Phase-1 scoping/acceptance artifact before approving execution.
7. During execution, use `/agents` to monitor the active hierarchy.
8. Keep canonical research-memory writes centralized through the Research Memory Keeper.
9. After the campaign, inspect `DECISION_MEMO.md` and the research-memory diffs.

If Teamwork is unavailable, run the campaign manually with the custom subagents from `.agents/agents/`, then ask `research-orchestrator` to synthesize the independent reports.
