"""
Transformer-Compatible Tokenizer — Wraps SciBERT tokenizer with special
handling for mathematical expressions ([MATH]...[/MATH] blocks).
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
from transformers import AutoTokenizer
from config import SCIBERT_MODEL, TRAINING_CONFIG


class MathAwareTokenizer:
    """
    Tokenizer that handles mathematical expressions as special tokens.
    Wraps HuggingFace AutoTokenizer with math-aware processing.
    """

    MATH_TAG_PATTERN = re.compile(r'\[MATH\](.*?)\[/MATH\]')

    def __init__(self, model_name: str = SCIBERT_MODEL, max_length: int = None):
        self.max_length = max_length or TRAINING_CONFIG["max_length"]
        print(f"[Phase 2] Loading tokenizer: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens for math
        special_tokens = {
            "additional_special_tokens": ["[MATH]", "[/MATH]", "[EQ]"]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"  ✓ Tokenizer loaded. Vocab size: {len(self.tokenizer)} (+{num_added} special tokens)")

    def tokenize(self, text: str, return_tensors: str = "pt") -> Dict:
        """
        Tokenize text with math-aware processing.
        [MATH]...[/MATH] blocks are treated as special token sequences.
        
        Returns:
            Dict with input_ids, attention_mask, and optionally token_type_ids
        """
        # Replace math blocks with [EQ] placeholder for transformer input
        processed_text = self.MATH_TAG_PATTERN.sub('[EQ]', text)
        
        encoding = self.tokenizer(
            processed_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )
        
        return encoding

    def tokenize_batch(self, texts: List[str], return_tensors: str = "pt") -> Dict:
        """Tokenize a batch of texts."""
        processed = [self.MATH_TAG_PATTERN.sub('[EQ]', t) for t in texts]
        
        encoding = self.tokenizer(
            processed,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )
        
        return encoding

    def decode(self, token_ids) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def get_vocab_size(self) -> int:
        """Get total vocabulary size including special tokens."""
        return len(self.tokenizer)

    def tokenize_for_analysis(self, text: str) -> Dict:
        """
        Tokenize and return detailed analysis including token count,
        math token positions, etc.
        """
        # Count math expressions
        math_matches = self.MATH_TAG_PATTERN.findall(text)
        
        encoding = self.tokenize(text)
        input_ids = encoding["input_ids"][0]
        
        # Find [EQ] token positions
        eq_token_id = self.tokenizer.convert_tokens_to_ids("[EQ]")
        eq_positions = (input_ids == eq_token_id).nonzero(as_tuple=True)[0].tolist()
        
        return {
            "encoding": encoding,
            "num_tokens": (encoding["attention_mask"][0] == 1).sum().item(),
            "num_math_expressions": len(math_matches),
            "math_token_positions": eq_positions,
            "truncated": (encoding["attention_mask"][0] == 1).sum().item() >= self.max_length,
        }


if __name__ == "__main__":
    tokenizer = MathAwareTokenizer()
    test = "The function [MATH]f(x) = x^2[/MATH] is continuous on [MATH]\\mathbb{R}[/MATH]."
    result = tokenizer.tokenize_for_analysis(test)
    print(f"\nTokens: {result['num_tokens']}")
    print(f"Math expressions: {result['num_math_expressions']}")
    print(f"Math positions: {result['math_token_positions']}")
    print(f"Truncated: {result['truncated']}")
