# game/ui_elements.py
import pygame as pg
from . import config # Import config from the same package

class Text(object):
    def __init__(self, textFontPath, size, message, color, xpos, ypos, center_x=False, center_y=False): # ADDED center_x, center_y
        try:
            self.font = pg.font.Font(textFontPath, size)
        except pg.error: 
            # print(f"Warning: Font '{textFontPath}' not found. Using default system font.") # Optional warning
            self.font = pg.font.Font(None, size) 
        
        self.surface = self.font.render(message, True, color) # True for antialiasing
        
        # Get current rect to calculate centered position if needed
        current_rect = self.surface.get_rect()
        new_topleft_x, new_topleft_y = xpos, ypos

        if center_x:
            new_topleft_x = xpos - current_rect.width // 2
        if center_y:
            new_topleft_y = ypos - current_rect.height // 2
        
        self.rect = self.surface.get_rect(topleft=(new_topleft_x, new_topleft_y))

    def draw(self, surface_to_draw_on):
        surface_to_draw_on.blit(self.surface, self.rect)

    # Optional: method to change text content or color and re-center if needed
    def update_text(self, message, color, center_x=None, center_y=None): 
        self.surface = self.font.render(message, True, color)
        
        # Preserve original centering intention if not overridden
        # This requires storing the original xpos, ypos, and centering flags or re-evaluating
        # For simplicity, let's assume re-centering based on current rect's topleft before this update
        # A more robust way would be to store original x,y and centering flags.
        
        old_topleft_x, old_topleft_y = self.rect.topleft
        current_rect = self.surface.get_rect() # New surface dimensions

        new_topleft_x, new_topleft_y = old_topleft_x, old_topleft_y # Default to old topleft

        # If centering was originally applied or newly requested, recalculate based on original x,y anchor
        # This part is tricky without storing initial xpos, ypos, and centering flags.
        # A simpler re-center based on the *current* center:
        old_center = self.rect.center 
        self.rect = self.surface.get_rect(center=old_center) # Re-center new surface at old center

        # If explicit centering is passed during update:
        if center_x is not None or center_y is not None:
            # This would ideally use the initial xpos, ypos anchor point, not self.rect.x/y
            # For now, let's assume if update_text is called, it might need to re-evaluate centering
            # based on some anchor. If just changing color/text on existing centered text,
            # the re-center on old_center above is usually fine.
            pass 