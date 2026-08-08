"""
tui/common.py — shared TUI constants (banner).
"""
BANNER = r"""
(()/(   )\    )\ )     )\())   ) (           ( /(
 /(_)|(((_)( (()/(  __((_)\ ( /( )(   (   (  )\())
(_))  )\ _ )\ /(_))|___|((_))(_)|()\  )\  )\((_)\
| _ \ (_)_\(_|_)) __| | \| ((_)_ ((_)((_)((_) |(_)
|   /  / _ \   | (_ | | .` / _` | '_/ _ Y _|| / /
|_|_\ /_/ \_\   \___| |_|\_\__,_|_| \___|__||_\_\
""".strip('\n').splitlines()
BANNER_H = len(BANNER)
BANNER_W = max(len(l) for l in BANNER)

__all__ = ['BANNER', 'BANNER_H', 'BANNER_W']
