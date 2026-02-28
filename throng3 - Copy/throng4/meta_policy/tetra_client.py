"""
Tetra Client — Interface to Tetra via OpenClaw Gateway.

Uses OpenClawBridge to communicate with Tetra through the gateway.
Maintains conversation history for multi-turn dialogue.
"""

from typing import Optional, List, Dict
from throng4.llm_policy.openclaw_bridge import OpenClawBridge


class TetraClient:
    """
    Client for querying Tetra via OpenClaw Gateway.
    
    Uses OpenClawBridge which handles gateway communication.
    """
    
    def __init__(self, game: str = "UnknownGame"):
        """
        Initialize Tetra client with OpenClaw bridge.
        
        Args:
            game: Game identifier for the bridge (can be generic)
        """
        self.bridge = OpenClawBridge(game=game)
        self.conversation_history: List[Dict[str, str]] = []
        self.episode_counter = 0
    
    def check_gateway(self) -> bool:
        """Check if gateway is available."""
        return self.bridge.check_gateway()
    
    def query(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Send prompt to Tetra and get response.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (not used by bridge, kept for compatibility)
            
        Returns:
            Tetra's text response
        """
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': prompt,
        })
        
        try:
            # Query Tetra via bridge
            response = self.bridge.query(prompt)
            
            # Extract response text
            tetra_response = response.raw if hasattr(response, 'raw') else str(response)
            
            # Add to history
            self.conversation_history.append({
                'role': 'assistant',
                'content': tetra_response,
            })
            
            return tetra_response
            
        except Exception as e:
            error_msg = f"Error querying Tetra via gateway: {e}"
            print(f"[TetraClient] {error_msg}")
            return error_msg
    
    def send_observation(self, observation: str, context: Optional[Dict] = None) -> str:
        """
        Send an observation to Tetra (for tracking, not expecting response).
        
        Args:
            observation: Observation text
            context: Optional context dict
            
        Returns:
            Tetra's acknowledgment
        """
        try:
            response = self.bridge.send_observation(
                episode=self.episode_counter,
                observation=observation,
                context=context or {}
            )
            
            return response.raw if hasattr(response, 'raw') else str(response)
            
        except Exception as e:
            error_msg = f"Error sending observation: {e}"
            print(f"[TetraClient] {error_msg}")
            return error_msg
    
    def reset_conversation(self):
        """Clear conversation history and increment episode."""
        self.conversation_history.clear()
        self.episode_counter += 1


if __name__ == "__main__":
    """Test Tetra client connection via gateway."""
    print("=" * 60)
    print("TETRA CLIENT TEST (via OpenClaw Gateway)")
    print("=" * 60)
    
    client = TetraClient(game="TestGame")
    
    print("\nChecking gateway connection...")
    if not client.check_gateway():
        print("⚠️  Gateway not available!")
        print("\nMake sure OpenClaw gateway is running:")
        print("  openclaw gateway start")
        print("  openclaw gateway health")
    else:
        print("✅ Gateway is available!")
        
        # Test query
        print("\nTesting query...")
        response = client.query("Hello, can you hear me?")
        
        if "Error" in response:
            print(f"⚠️  {response}")
        else:
            print(f"✅ Tetra responded: {response[:100]}...")
            
            # Test multi-turn
            print("\nTesting multi-turn dialogue...")
            response2 = client.query("What did I just say?")
            print(f"✅ Tetra responded: {response2[:100]}...")
    
    print("\n" + "=" * 60)
