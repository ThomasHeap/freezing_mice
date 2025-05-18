from pydantic import BaseModel, Field

class MouseBehaviorSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in HRS:MIN:SEC")
    end_time: str = Field(..., description="End time in HRS:MIN:SEC")
    behavior: str = Field(..., description="Behavior label (e.g., attack, investigation, mount, other)") 