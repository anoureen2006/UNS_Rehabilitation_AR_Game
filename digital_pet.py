"""
digital_pet.py
Defines the Digital Pet object placed by the RL agent in the AR Rehabilitation room.
Handles coordinate/cell ID conversions, valid placement checks against obstacles,
and proximity/reach detection.
"""

import numpy as np
from config import ROOM_WIDTH, ROOM_HEIGHT, GRID_COLS, GRID_ROWS, CELL_WIDTH, CELL_HEIGHT, NUM_CELLS

class DigitalPet:
    """
    Virtual AR Digital Pet object.
    The RL agent selects the pet's position (cell ID 0..399 or (x,y) coordinates)
    to motivate patient spatial exploration.
    """
    def __init__(self, init_cell=None):
        if init_cell is None:
            init_cell = 0  # Top-left cell default
        self.cell_id = init_cell
        self.pos = self.cell_id_to_coords(self.cell_id)
        self.is_visible = True

    @staticmethod
    def cell_id_to_coords(cell_id):
        """Converts integer cell ID (0 to 399) to (x, y) continuous room coordinates (cell center)."""
        cell_id = int(np.clip(cell_id, 0, NUM_CELLS - 1))
        row = cell_id // GRID_COLS
        col = cell_id % GRID_COLS
        x = (col + 0.5) * CELL_WIDTH
        y = (row + 0.5) * CELL_HEIGHT
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def coords_to_cell_id(coords):
        """Converts continuous (x, y) coordinates to grid cell ID (0 to 399)."""
        col = int(np.clip(coords[0] / CELL_WIDTH, 0, GRID_COLS - 1))
        row = int(np.clip(coords[1] / CELL_HEIGHT, 0, GRID_ROWS - 1))
        return row * GRID_COLS + col

    def set_position(self, target):
        """
        Sets pet position from either an integer cell ID (0..399) or
        a 2D numpy array/tuple [x, y].
        """
        if isinstance(target, (int, np.integer)):
            self.cell_id = int(target)
            self.pos = self.cell_id_to_coords(self.cell_id)
        else:
            self.pos = np.array(target, dtype=np.float32)
            self.pos[0] = np.clip(self.pos[0], 0.2, ROOM_WIDTH - 0.2)
            self.pos[1] = np.clip(self.pos[1], 0.2, ROOM_HEIGHT - 0.2)
            self.cell_id = self.coords_to_cell_id(self.pos)

    def is_reached(self, patient_pos, reach_radius=0.8):
        """Checks if the patient is within reach radius of the pet."""
        dist = np.linalg.norm(self.pos - patient_pos)
        return dist <= reach_radius

    @staticmethod
    def is_valid_location(pos, obstacles=None):
        """Checks if the pet position is clear of obstacles."""
        if pos[0] < 0.3 or pos[0] > (ROOM_WIDTH - 0.3) or pos[1] < 0.3 or pos[1] > (ROOM_HEIGHT - 0.3):
            return False
            
        if obstacles is not None:
            for obs in obstacles:
                dist = np.linalg.norm(pos - obs['pos'])
                if dist < (obs['radius'] + 0.3):
                    return False
        return True
