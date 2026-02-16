# TODO: API Key Safety Setup

**Priority:** Do after LLM policy decomposition is planned/scaffolded

## Plan
1. Create `.env` file for local API keys (add to `.gitignore`)
2. Create `.env.example` template
3. Create `throng4/config/api_config.py` — safe key loader using `python-dotenv`
4. Ensure keys never appear in logs, error messages, or client-side code
5. Consider OpenClaw's built-in token management if we adopt it

## OpenClaw Integration Note
OpenClaw already provides agentic access, subagent management, and token/API access
to multiple LLM models. If adopted, its secret management may supersede `.env`-based
approach. Evaluate after scaffolding the policy decomposition system.
