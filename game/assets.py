# game/assets.py
import pygame as pg
import os
from . import config # Import config from the same package

IMAGES = {}
_images_converted_in_process = {} # Tracks if IMAGES for a process are converted (PID -> bool)

class DummySound:
    def play(self, *args, **kwargs): pass
    def stop(self, *args, **kwargs): pass
    def fadeout(self, *args, **kwargs): pass
    def set_volume(self, *args, **kwargs): pass

def load_all_game_images():
    """
    Loads all game images into the global IMAGES dictionary.
    Attempts to convert_alpha() them. This function assumes that
    pygame.display has been initialized and a mode has been set
    (even if it's a dummy one for headless workers).
    """
    global IMAGES, _images_converted_in_process
    pid = os.getpid()

    if _images_converted_in_process.get(pid, False):
        return

    can_convert = False
    if pg.display.get_init():
        if pg.display.get_surface() is not None:
            can_convert = True
        else:
            try:
                if not pg.display.get_init(): pg.init() 
                pg.display.set_mode((1,1), pg.NOFRAME) 
                can_convert = True
            except pg.error as e:
                print(f"PID {pid}: CRITICAL - Failed to set minimal dummy display: {e}. Images not converted.", flush=True)
                can_convert = False
    else:
        print(f"PID {pid}: Pygame display not initialized before image load. Images not converted.", flush=True)

    for name in config.IMG_NAMES:
        try:
            image_path = os.path.join(config.IMAGE_PATH, f'{name}.png')
            loaded_image = pg.image.load(image_path)
            if can_convert:
                IMAGES[name] = loaded_image.convert_alpha()
            else:
                IMAGES[name] = loaded_image 
        except pg.error as e_load:
            print(f"PID {pid}: Error loading image '{image_path}': {e_load}", flush=True)
            IMAGES[name] = pg.Surface((30, 30)); IMAGES[name].fill(config.RED)
    
    _images_converted_in_process[pid] = True