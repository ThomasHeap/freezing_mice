def seconds_to_hms(seconds):
    """Convert seconds to HRS:MIN:SEC format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def hms_to_seconds(hms_str):
    """Convert HRS:MIN:SEC string to seconds"""
    if isinstance(hms_str, float):
        return hms_str
    h, m, s = map(int, hms_str.split(':'))
    return h * 3600 + m * 60 + s 