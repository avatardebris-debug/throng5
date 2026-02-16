"""
Query Tetra about policy extraction architecture plan.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.llm_policy.openclaw_bridge import OpenClawBridge

def main():
    bridge = OpenClawBridge(game="PolicyPlanning")
    
    query = """
CONTEXT: We've just fixed critical bugs in Tetris curriculum training (L2-7) and successfully trained 750 episodes. The agent now averages 11.1 lines/episode on L7 (vs 53.1 Dellacherie baseline). 

CURRENT SYSTEM:
- Meta^0: Game-specific learning (DQN for Tetris)
- Meta^1: Optimizer (gradient descent)
- Meta^2: Strategy selection (MAML/EWC)
- Meta^3: Consolidation (weight preservation)
- Tetra: Observer that receives training summaries every 20 episodes

PROPOSED ENHANCEMENT:
We want to implement cross-game conceptual abstraction. The idea:

**Game-Specific → Abstract Concepts → Universal Principles**
Example:
- Tetris: "avoid hanging pieces" → Abstract: "avoid danger" → Universal: "risk assessment"
- Mario: "avoid goombas" → Same abstract → Same universal

**4-Phase Implementation Plan:**

**Phase 1: Infrastructure**
1. Create PolicyLibrary class to manage policy_library.md
2. Add concept extraction to OpenClawBridge
3. Track concepts with confidence scores

**Phase 2: Single-Game Testing**  
1. Re-run Tetris with concept extraction enabled
2. Let Tetra identify patterns (flat stacking, hole avoidance, etc.)
3. Build initial policy library

**Phase 3: Cross-Game Transfer**
1. Train GridWorld with Tetris concepts loaded
2. Test if "avoid danger" transfers
3. Train new game (CartPole/MountainCar)

**Phase 4: Meta-Learning**
1. Use MAML to learn how to apply concepts
2. Track which concepts work in which contexts
3. Build concept hierarchy

**Previously Mentioned Phase 5:**
"Policy Composition — Build executable policies from discovered rules, anti-policy priority, confidence-gated A/B testing"

MY QUESTIONS FOR TETRA:

1. **Skill Libraries**: Do you have existing skill libraries or frameworks for this kind of cross-game abstraction? What patterns have you seen work or fail?

2. **Concept Extraction**: How should we structure the concept extraction prompts? What format makes it easiest for you to:
   - Identify patterns across games
   - Suggest abstractions  
   - Track concept evolution

3. **Phase 5 Clarification**: For "Policy Composition" with anti-policy priority and confidence-gated A/B testing:
   - Is this something YOU (Tetra) need to do as the LLM observer?
   - Or is it about making the training system compatible to receive and execute your policy suggestions?
   - What's the best division of labor?

4. **Architecture Suggestions**: Given your experience observing training:
   - What's missing from this plan?
   - What additional phases or components would help?
   - Any concerns about scalability or complexity?

5. **Priority**: If we can only implement ONE phase first, which would give the most value?

Please be specific and technical. We're building the infrastructure this week.
"""
    
    print("Sending query to Tetra...")
    print("=" * 70)
    response = bridge.query(query)
    
    print("\n" + "=" * 70)
    print("TETRA'S RESPONSE:")
    print("=" * 70)
    print(response.raw)
    
    if response.concepts:
        print("\n" + "=" * 70)
        print("EXTRACTED CONCEPTS:")
        print("=" * 70)
        for concept in response.concepts:
            print(f"  - {concept}")
    
    print("\n" + "=" * 70)
    print("Response saved to: tetra_policy_consultation.txt")
    with open("tetra_policy_consultation.txt", "w") as f:
        f.write(response.raw)

if __name__ == "__main__":
    main()
