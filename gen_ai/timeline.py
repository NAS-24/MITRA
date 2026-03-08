from datetime import datetime, timedelta

class TimelineManager:
    def __init__(self, debounce_minutes=15):
        # debounce_minutes: Time window to treat multiple sightings as one session
        self.debounce_minutes = debounce_minutes

    def get_time_ago(self, last_seen_str):
        """Converts a timestamp into a human-friendly string."""
        if not last_seen_str:
            return "the first time"
        
        last_seen = datetime.fromisoformat(last_seen_str)
        now = datetime.now()
        diff = now - last_seen

        if diff < timedelta(minutes=60):
            return f"{int(diff.seconds / 60)} minutes ago"
        elif diff < timedelta(days=1):
            return f"{int(diff.seconds / 3600)} hours ago"
        else:
            return f"{diff.days} days ago"

    def should_narrate(self, person_id, last_narration_time):
        """Prevents the system from speaking every time the face is seen."""
        if not last_narration_time:
            return True
        
        last_time = datetime.fromisoformat(last_narration_time)
        return (datetime.now() - last_time) > timedelta(minutes=self.debounce_minutes)

# --- Test with Mock Data ---
if __name__ == "__main__":
    manager = TimelineManager()
    
    # Example: Last seen 2 hours ago
    mock_timestamp = (datetime.now() - timedelta(hours=2)).isoformat()
    
    print(f"Human-friendly time: {manager.get_time_ago(mock_timestamp)}")
    print(f"Should MITRA speak again? {manager.should_narrate('user_101', mock_timestamp)}")