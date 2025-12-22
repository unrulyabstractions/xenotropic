"""
String basics: Creating and manipulating strings.

This example demonstrates the core String abstraction in xenotechnics.
"""

from xenotechnics.common import String

print("=" * 60)
print("STRING BASICS")
print("=" * 60)

# ============================================================================
# 1. Creating Strings
# ============================================================================

print("\n1. CREATING STRINGS")
print("-" * 60)

# From text (single token)
s1 = String.from_text("hello")
print(f"From text (single token):  {s1.tokens}")
print(f"  -> String: {s1}")
print(f"  -> Length: {len(s1)}")

# From tokens directly
s2 = String.from_tokens(("H", "e", "l", "l", "o"))
print(f"\nFrom tokens:               {s2.tokens}")
print(f"  -> Length: {len(s2)}")

# Manual construction
s3 = String(tokens=("w", "o", "r", "l", "d"))
print(f"\nManual construction:       {s3.tokens}")

# Empty string (for tree root)
empty = String.empty()
print(f"\nEmpty string:              {empty.tokens}")
print(f"  -> Length: {len(empty)}")

# ============================================================================
# 2. Prompt vs Continuation
# ============================================================================

print("\n\n2. PROMPT VS CONTINUATION")
print("-" * 60)

# Create string with explicit prompt length
s_with_prompt = String(
    tokens=("The", " ", "sky", " ", "is", " ", "blue"),
    prompt_length=5  # "The", " ", "sky", " ", "is" is prompt
)
print(f"Full tokens:           {s_with_prompt.tokens}")
print(f"Prompt length:         {s_with_prompt.prompt_length}")
print(f"Prompt tokens:         {s_with_prompt.prompt_tokens()}")
print(f"Continuation tokens:   {s_with_prompt.continuation_tokens()}")

# Extend preserves prompt length
extended = s_with_prompt.extend("!")
print(f"\nAfter extend('!'):")
print(f"  Full tokens:         {extended.tokens}")
print(f"  Prompt tokens:       {extended.prompt_tokens()}")
print(f"  Continuation:        {extended.continuation_tokens()}")

# ============================================================================
# 3. String Operations
# ============================================================================

print("\n\n3. STRING OPERATIONS")
print("-" * 60)

base = String.from_tokens(("c", "a", "t"))
print(f"Base string: {base.tokens}")

# Extend (immutable - creates new string)
extended = base.extend("s")
print(f"\nAfter extend('s'):      {extended.tokens}")
print(f"Original unchanged:     {base.tokens}")

# Chain extensions
result = base.extend(" ").extend("a").extend("n").extend("d")
print(f"\nChained extensions:     {result.tokens}")

# ============================================================================
# 4. Text Representation
# ============================================================================

print("\n\n4. TEXT REPRESENTATION")
print("-" * 60)

s = String.from_tokens(("H", "e", "l", "l", "o", "!"))

# Default to_text
print(f"to_text():             '{s.to_text()}'")

# With separator
print(f"to_text('|'):          '{s.to_text(separator='|')}'")

# __str__ method
print(f"str(s):                '{str(s)}'")

# ============================================================================
# 5. Token IDs (for LLM integration)
# ============================================================================

print("\n\n5. TOKEN IDs (LLM INTEGRATION)")
print("-" * 60)

# String with token IDs (simulated)
s_with_ids = String(
    tokens=("hello", " ", "world"),
    token_ids=(42, 3, 99)
)
print(f"Tokens:    {s_with_ids.tokens}")
print(f"Token IDs: {s_with_ids.token_ids}")

# Extend with token ID
extended_with_id = s_with_ids.extend_with_token_id("!", 100)
print(f"\nAfter extend with ID:")
print(f"  Tokens:    {extended_with_id.tokens}")
print(f"  Token IDs: {extended_with_id.token_ids}")

# ============================================================================
# 6. Practical Pattern: Building Generation Sequences
# ============================================================================

print("\n\n6. PRACTICAL: BUILDING GENERATION")
print("-" * 60)

# Start with prompt (as tokens from tokenizer)
prompt_tokens = ("Once", " upon", " a", " time")
prompt = String.from_tokens(prompt_tokens)
print(f"Prompt: {prompt.tokens}")

# Set prompt length to mark where generation starts
current = String(tokens=prompt.tokens, prompt_length=len(prompt))
print(f"Prompt length: {current.prompt_length}")

# Simulate generation
tokens_to_generate = [" there", " was", " a"]
for token in tokens_to_generate:
    current = current.extend(token)
    print(f"  Generated: {current.continuation_tokens()}")

print(f"\nFinal string:")
print(f"  Prompt:       {current.prompt_tokens()}")
print(f"  Generated:    {current.continuation_tokens()}")
print(f"  Full text:    '{current.to_text()}'")

print("\n" + "=" * 60)
print("STRING EXAMPLES COMPLETE")
print("=" * 60)
