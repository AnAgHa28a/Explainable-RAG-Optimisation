from src.prompting.prompt_generator import (
    build_basic_prompt,
    build_context_prompt,
    build_cot_prompt
)

query = "The movie was fantastic"

context = "Great acting and wonderful film"

retrieved = [
    "Amazing movie with great acting",
    "I loved this film",
    "Worst movie ever made"
]

print("\n--- BASIC ---\n")
print(build_basic_prompt(query, context))

print("\n--- CONTEXT ---\n")
print(build_context_prompt(query, retrieved))

print("\n--- COT ---\n")
print(build_cot_prompt(query, context))