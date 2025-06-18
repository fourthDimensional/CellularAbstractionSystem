import sys

import pygame


class SimulationEngine:
    def __init__(self):
        self.running = True

        pass

    def run(self):
        while self.running:
            self._handle_frame()

        pygame.quit()
        sys.exit()

    def _handle_frame(self):
        pass