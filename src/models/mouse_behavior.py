from pydantic import BaseModel, Field

class MouseBehaviorSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: str = Field(..., description="End time in MM:SS format")
    behavior: str = Field(..., description="Behavior label (e.g., attack, investigation, mount, other)")
    
class ScratchAidSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: str = Field(..., description="End time in MM:SS format")
    behavior: str = Field(..., description="Behavior label (e.g., scratching, not scratching)")
    
class MouseBoxSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: str = Field(..., description="End time in MM:SS format")
    behavior: str = Field(..., description="Behavior label (e.g., bedding box, other)")
    
class GroomingSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: str = Field(..., description="End time in MM:SS format")
    behavior: str = Field(..., description="Behavior label (e.g., grooming, not grooming)")
    
class FreezingSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: str = Field(..., description="End time in MM:SS format")
    behavior: str = Field(..., description="Behavior label (e.g., Freezing, Not Freezing)")
    
class ForagingSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: str = Field(..., description="End time in MM:SS format")
    behavior: str = Field(..., description="Behavior label (e.g., foraging, not foraging)")
    
class MouseVentralSegment(BaseModel):
    segment_number: int = Field(..., description="Segment number in order")
    start_time: str = Field(..., description="Start time in MM:SS format")
    end_time: str = Field(..., description="End time in MM:SS format")
    behavior: str = Field(..., description="Behavior label (e.g., background, scratch, lick)")  