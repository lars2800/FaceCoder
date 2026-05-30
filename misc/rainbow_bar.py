import tqdm
import math

#
# Yay custimization!
#

class RainbowProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def get_rainbow_rgb(phase):
        """Generates vibrant RGB values covering all main colors using HSV."""
        x = (phase % 1.0) * math.pi * math.pi * 0.5 # i love me some magic numbers!
        
        g_float = max( math.sin( x  ), 0 )
        b_float = max( math.sin( x - 0.5 * math.pi ), 0)
        r_float= max( math.sin( x + 0.5 * math.pi ), 0)
        
        # Convert from [0.0, 1.0] floats to [0, 255] integers
        r = int(r_float * 255)
        g = int(g_float * 255)
        b = int(b_float * 255)
        
        return r, g, b

    @classmethod
    def format_meter(cls, n, total, elapsed, *args, **kwargs):
        # 1. Get the standard formatted string components from tqdm
        meter_str = super().format_meter(n, total, elapsed, *args, **kwargs)
        
        if not total:
            return meter_str
            
        filled_char = '█'
        
        # If the bar hasn't started filling yet, just return the standard string
        if filled_char not in meter_str:
            return meter_str
            
        # 2. Isolate the portions of the bar
        parts = meter_str.split(filled_char)
        num_filled = meter_str.count(filled_char)
        
        # 3. Determine the total maximum width of the bar.
        # We can calculate this by looking at how many total block spaces (filled + empty) 
        # exist in the bar section of the string.
        # tqdm uses spaces ' ' or light blocks '░' for the remaining bar.
        # A robust way is to check the length of the bar segment using tqdm's internal width logic:
        dynamic_ncols = kwargs.get('ncols') or 40 # Fallback default width if undetected
        
        # If a custom bar_format is used (like {bar:50}), we grab that specific width
        bar_format = kwargs.get('bar_format')
        max_bar_width = dynamic_ncols
        if bar_format and '{bar:' in bar_format:
            try:
                max_bar_width = int(bar_format.split('{bar:')[1].split('}')[0])
            except ValueError:
                pass

        # 4. Rebuild the filled portion of the bar
        colored_bar = ""
        for i in range(num_filled):
            # Crucial Change: Divide by (max_bar_width - 1) instead of num_filled.
            # This locks the color to its absolute horizontal position on the screen.
            phase = i / (max_bar_width - 1) if max_bar_width > 1 else 0.0
            
            r, g, b = cls.get_rainbow_rgb(phase)
            colored_bar += f"\033[38;2;{r};{g};{b}m{filled_char}"
            
        # Reset ANSI terminal colors
        colored_bar += "\033[0m"
        
        # 5. Splice our revealed rainbow back into the meter layout
        prefix = parts[0]
        suffix = parts[-1]
        
        return f"{prefix}{colored_bar}{suffix}"
