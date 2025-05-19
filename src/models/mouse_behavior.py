from pydantic import BaseModel, Field

class MouseBehaviorSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: float = Field(..., description="Start time in SEC")
    end_time: float = Field(..., description="End time in SEC")
    behavior: str = Field(..., description="Behavior label (e.g., attack, investigation, mount, other)")
    
class ScratchAidSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: float = Field(..., description="Start time in SEC")
    end_time: float = Field(..., description="End time in SEC")
    behavior: str = Field(..., description="Behavior label (e.g., scratching)")