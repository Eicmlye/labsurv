import time

def get_time_stamp():
  cur_time = time.localtime()
  return (
    f"{cur_time.tm_year % 100}{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}"
    f"_{cur_time.tm_hour:02d}{cur_time.tm_min:02d}{cur_time.tm_sec:02d}"
  )