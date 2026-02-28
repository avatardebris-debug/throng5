import sys
sys.path.insert(0, '.')
from throng4.llm_policy.openclaw_bridge import OpenClawBridge

bridge = OpenClawBridge(game="ALE/Breakout-v5")

prompt = (
    "You are reviewing a compressed gameplay log for Breakout.\n"
    "Each line is: Step | Action | State Variables | Reward | Lives.\n\n"
    "Analyze the log and return a JSON block with this EXACT structure:\n"
    "{\"hypotheses\": [{\"id\": \"rule_paddle_align\", \"description\": \"Keep paddle aligned under ball\","
    " \"object\": \"paddle\", \"feature\": \"paddle_x\", \"direction\": \"maximize\","
    " \"trigger\": \"ball approaching paddle\", \"confidence\": 0.7}]}\n\n"
    "### GAME LOG ###\n"
    "Step 001 | Action: Fire   | Paddle_X: 072, Ball: (195, 179) | Reward: 0.0 | Lives: 5\n"
    "Step 050 | Action: Right  | Paddle_X: 086, Ball: (195, 179) | Reward: 0.0 | Lives: 2\n"
    "Step 100 | Action: Left   | Paddle_X: 098, Ball: (079, 100) | Reward: 1.0 | Lives: 2\n"
)

r = bridge.send_observation(episode=1, observation=prompt)
print("SUCCESS:", r.success)
print("ERROR:", r.error)
print("HYPOTHESES count:", len(r.hypotheses))
if r.hypotheses:
    print("FIRST HYPOTHESIS:", r.hypotheses[0])
print("RAW (first 2000 chars):")
print(r.raw[:2000])
