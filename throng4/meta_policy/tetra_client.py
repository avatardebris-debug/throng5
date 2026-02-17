"""
Tetra Client — Interface to Tetra LLM for hypothesis generation.

Simple HTTP wrapper for querying Tetra's API.
Maintains conversation history for multi-turn dialogue.
"""

import requests
from typing import Optional, List, Dict


class TetraClient:
    """
    Simple client for querying Tetra.
    
    Tetra should be running locally (e.g., on port 8000).
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.conversation_history: List[Dict[str, str]] = []
    
    def query(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Send prompt to Tetra and get response.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            
        Returns:
            Tetra's text response
        """
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': prompt,
        })
        
        # Query Tetra API
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={
                    'messages': self.conversation_history,
                    'temperature': temperature,
                },
                timeout=30,
            )
            response.raise_for_status()
            
            result = response.json()
            tetra_response = result.get('response', '')
            
            # Add to history
            self.conversation_history.append({
                'role': 'assistant',
                'content': tetra_response,
            })
            
            return tetra_response
            
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Tetra. Is it running?"
        except requests.exceptions.Timeout:
            return "Error: Tetra query timed out."
        except Exception as e:
            print(f"[TetraClient] Error: {e}")
            return f"Error querying Tetra: {e}"
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()


if __name__ == "__main__":
    """Test Tetra client connection."""
    print("=" * 60)
    print("TETRA CLIENT TEST")
    print("=" * 60)
    
    client = TetraClient()
    
    print("\nTesting connection to Tetra...")
    response = client.query("Hello, can you hear me?")
    
    if "Error" in response:
        print(f"⚠️  {response}")
        print("\nMake sure Tetra is running on http://localhost:8000")
    else:
        print(f"✅ Tetra responded: {response[:100]}...")
        
        # Test multi-turn
        print("\nTesting multi-turn dialogue...")
        response2 = client.query("What did I just say?")
        print(f"✅ Tetra responded: {response2[:100]}...")
    
    print("\n" + "=" * 60)
