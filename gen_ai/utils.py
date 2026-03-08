# gen_ai/utils.py
from timeline import TimelineManager

class ContextAssembler:
    def __init__(self):
        self.timeline_manager = TimelineManager()

    def prepare_llm_payload(self, backend_data):
        """
        Processes raw backend JSON and formats it for the GenAI Prompt.
        """
        # 1. Handle Unknown faces
        if backend_data.get("is_unknown"):
            return {"status": "unknown", "context": None}

        # 2. Handle Low Confidence (Human-in-the-loop trigger)
        confidence = backend_data.get("confidence_score", 0.0)
        if confidence < 0.7:
            return {
                "status": "needs_confirmation",
                "name_guess": backend_data.get("name"),
                "context": None
            }

        # 3. Calculate human-friendly time using your TimelineManager
        last_met_iso = backend_data.get("last_met_timestamp")
        relative_time = self.timeline_manager.get_time_ago(last_met_iso)

        # 4. Assemble the clean package for the LLM
        clean_context = {
            "name": backend_data.get("name"),
            "relation": backend_data.get("relationship"),
            "last_met": relative_time,
            "notes": backend_data.get("notes")
        }
        
        return {
            "status": "ready_for_llm",
            "context": clean_context
        }

# --- Quick Test ---
if __name__ == "__main__":
    from mock_data import MOCK_HIGH_CONFIDENCE, MOCK_LOW_CONFIDENCE
    
    assembler = ContextAssembler()
    
    print("Testing High Confidence:")
    print(assembler.prepare_llm_payload(MOCK_HIGH_CONFIDENCE))
    
    print("\nTesting Low Confidence:")
    print(assembler.prepare_llm_payload(MOCK_LOW_CONFIDENCE))