# gen_ai/prompts.py

class MitraPrompts:
    """Stores the carefully engineered prompts for the LLM."""

    @staticmethod
    def get_system_persona(target_language="English"):
        return (
            f"You are MITRA, a warm, patient memory assistant for someone experiencing memory loss. "
            f"CRITICAL RULES: "
            f"1. Keep it brief: Maximum 2 short sentences. "
            f"2. ALWAYS start your response directly with 'This is [Name]...' (translated to the target language if necessary). "
            f"3. Do not add any introductory filler or greetings like 'Hello' or 'I can help'. "
            f"4. Be comforting, but STRICTLY FACTUAL. "
            f"5. NEVER suggest future activities, make plans, or assume what the person wants to do right now. "
            f"6. NEVER mention 'databases', 'systems', 'AI', or 'face recognition'. "
            f"7. You MUST output your final spoken response entirely in {target_language}."
        )

    @staticmethod
    def get_narration_prompt(context):
        """Builds the prompt for a high-confidence recognition."""
        return (
            f"Here are the facts about the person standing in front of the user:\n"
            f"- Name: {context.get('name')}\n"
            f"- Relation: {context.get('relation')}\n"
            f"- Last met: {context.get('last_met')}\n"
            f"- Helpful reminder: {context.get('notes')}\n\n"
            f"Generate what I should say to the user."
        )

    @staticmethod
    def get_confirmation_text(name_guess, target_language="English"):
        # Simple hardcoded fallbacks for speed
        if target_language.lower() == "hindi":
            return f"मुझे लगता है कि यह {name_guess} है। क्या मैं सही हूँ?"
        return f"I think this might be {name_guess}. Am I right?"

# --- Quick Test ---
if __name__ == "__main__":
    from utils import ContextAssembler
    from mock_data import MOCK_HIGH_CONFIDENCE
    
    # Simulate processing the backend data
    assembler = ContextAssembler()
    payload = assembler.prepare_llm_payload(MOCK_HIGH_CONFIDENCE)
    
    # Generate the prompt
    prompts = MitraPrompts()
    print("=== SYSTEM INSTRUCTIONS ===")
    print(prompts.get_system_persona())
    print("\n=== DATA SENT TO LLM ===")
    print(prompts.get_narration_prompt(payload['context']))