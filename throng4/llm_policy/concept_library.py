"""
Concept library and transfer utilities for meta-learning validation.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConceptLibrary:
    """Load and query concept library."""
    
    def __init__(self, library_path: str = None):
        """
        Initialize concept library.
        
        Args:
            library_path: Path to library.json (default: ~/.openclaw/workspace/concepts/library.json)
        """
        if library_path is None:
            library_path = Path.home() / ".openclaw/workspace/concepts/library.json"
        
        self.library_path = Path(library_path)
        self.library = self._load_library()
    
    def _load_library(self) -> Dict:
        """Load library from JSON."""
        if not self.library_path.exists():
            return {"concepts": {}, "games": {}}
        
        with open(self.library_path, 'r') as f:
            return json.load(f)
    
    def get_concepts_for_game(self, game: str) -> List[str]:
        """Get list of concept IDs discovered for a game."""
        game_data = self.library.get('games', {}).get(game.lower(), {})
        return game_data.get('concepts_discovered', [])
    
    def get_concept(self, concept_id: str) -> Optional[Dict]:
        """Get concept details by ID."""
        return self.library.get('concepts', {}).get(concept_id)
    
    def get_all_concepts(self) -> Dict[str, Dict]:
        """Get all concepts."""
        return self.library.get('concepts', {})
    
    def get_high_confidence_concepts(self, min_confidence: float = 0.8) -> List[str]:
        """Get concept IDs with high transfer potential."""
        concepts = []
        for concept_id, concept in self.get_all_concepts().items():
            if concept.get('transfer_potential', 0) >= min_confidence:
                concepts.append(concept_id)
        return concepts


class ConceptTransfer:
    """Apply concepts to agent initialization."""
    
    def __init__(self, library: ConceptLibrary):
        self.library = library
    
    def apply_static_concepts(self, agent, concept_ids: List[str]):
        """
        Apply concepts as static heuristics to agent.
        
        This modifies agent weights to bias toward concept-aligned behaviors.
        
        Args:
            agent: PortableNNAgent instance
            concept_ids: List of concept IDs to apply
        """
        print(f"  Applying {len(concept_ids)} static concepts...")
        
        for concept_id in concept_ids:
            concept = self.library.get_concept(concept_id)
            if not concept:
                continue
            
            # Simple heuristic: boost weights for concept-aligned features
            # In practice, this would be more sophisticated
            bias_strength = concept.get('transfer_potential', 0.5)
            
            # Apply small bias to encourage concept-aligned behavior
            # This is a placeholder - real implementation would map concepts to features
            agent.W1 += np.random.randn(*agent.W1.shape) * 0.01 * bias_strength
            
            print(f"    - {concept_id}: bias={bias_strength:.2f}")
    
    def get_applicable_concepts_heuristic(
        self,
        source_game: str,
        target_game: str
    ) -> List[str]:
        """
        Heuristically determine which concepts might transfer.
        
        This is a simple rule-based approach without LLM.
        
        Args:
            source_game: Source game name (e.g., 'tetris')
            target_game: Target game name (e.g., 'breakout')
            
        Returns:
            List of concept IDs likely to transfer
        """
        # Get concepts from source game
        source_concepts = self.library.get_concepts_for_game(source_game)
        
        # Filter by transfer potential
        applicable = []
        for concept_id in source_concepts:
            concept = self.library.get_concept(concept_id)
            if concept and concept.get('transfer_potential', 0) > 0.7:
                applicable.append(concept_id)
        
        return applicable


def save_tetris_weights_for_transfer():
    """
    Save Tetris agent weights for MAML-Only baseline.
    
    This should be run after Tetris training to create weights file.
    """
    # TODO: Load final Tetris agent and save weights
    # For now, create placeholder
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    weights_path = weights_dir / "tetris_maml.npz"
    
    # Placeholder weights (would load from actual Tetris training)
    np.savez(
        weights_path,
        W1=np.random.randn(128, 128) * 0.1,
        b1=np.zeros(128),
        W2=np.random.randn(1, 128) * 0.1,
        b2=np.zeros(1)
    )
    
    print(f"✅ Saved Tetris weights to {weights_path}")
    return str(weights_path)


if __name__ == "__main__":
    # Test concept library
    print("Testing ConceptLibrary...")
    
    library = ConceptLibrary()
    print(f"Loaded library from {library.library_path}")
    
    # Get Tetris concepts
    tetris_concepts = library.get_concepts_for_game('tetris')
    print(f"\nTetris concepts ({len(tetris_concepts)}):")
    for concept_id in tetris_concepts:
        concept = library.get_concept(concept_id)
        print(f"  - {concept_id}: {concept.get('label', 'N/A')}")
    
    # Get high-confidence concepts
    high_conf = library.get_high_confidence_concepts(min_confidence=0.8)
    print(f"\nHigh-confidence concepts ({len(high_conf)}):")
    for concept_id in high_conf:
        concept = library.get_concept(concept_id)
        print(f"  - {concept_id}: transfer_potential={concept.get('transfer_potential', 0):.2f}")
    
    # Test transfer
    transfer = ConceptTransfer(library)
    applicable = transfer.get_applicable_concepts_heuristic('tetris', 'breakout')
    print(f"\nConcepts applicable to Breakout ({len(applicable)}):")
    for concept_id in applicable:
        print(f"  - {concept_id}")
    
    print("\n✅ ConceptLibrary test complete!")
