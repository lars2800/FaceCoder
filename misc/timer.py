from __future__ import annotations
import math
import time

class Timer:
    def __init__(self) -> None:
        pass

    def _start(self) -> None:
        self.start_time = time.time()
    
    def read(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        end_time = time.time()
        dt = end_time - self.start_time
        
        time_ms_total = math.floor( dt * 1000 )
        time_s_total  = math.floor( dt )
        time_m_total  = math.floor( time_s_total / 60 )
        time_h_total  = math.floor( time_m_total / 60 )

        time_ms = time_ms_total - ( time_s_total * 1000 ) - ( time_m_total * 60 * 1000 ) - ( time_h_total * 60 * 60 * 1000 )
        time_s  = time_s_total  - ( time_m_total * 60  ) - ( time_h_total * 60 * 60 )
        time_m  = time_m_total  - ( time_h_total * 60 )
        time_h  = time_h_total

        if ( time_h != 0 ):
            return f"{time_h}:{time_m} {time_s}.{time_ms}s"
        
        elif ( time_m != 0 ):
            return f"{time_m} min. {time_s}.{time_ms}s"
        
        elif ( time_s != 0 ):
            return f"{time_s}.{time_ms}s"
        
        elif ( time_ms != 0 ):
            return f"{time_ms} ms"

        else:
            return f"dt = 0"
    
    @staticmethod
    def begin() -> Timer:
        t = Timer()
        t._start()
        return t