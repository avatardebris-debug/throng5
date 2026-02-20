import sys
sys.path.insert(0, '.')
from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.llm_policy.eval_auditor import EvalAuditor

adapter = TetrisAdapter(level=2, seed=123)
adapter.reset()

config = AgentConfig(n_hidden=48, epsilon=0.0)
agent = PortableNNAgent(n_features=adapter.n_features, config=config, seed=123)

auditor = EvalAuditor()

actions_taken = []
reported_reward = 0.0
reported_lines = 0

for _ in range(15):
    valid_actions = adapter.get_valid_actions()
    if not valid_actions: break
    action = agent.select_action(valid_actions, adapter.make_features, explore=False)
    _, reward, done, info = adapter.step(action)
    actions_taken.append(action)
    reported_reward += reward
    reported_lines = info['lines_cleared']

audit_adapter = TetrisAdapter(level=2, seed=123)
report = auditor.audit_episode(1, 2, {'lines': reported_lines, 'reward': reported_reward}, actions_taken, audit_adapter)

print("Anomalies:", report.anomalies)
