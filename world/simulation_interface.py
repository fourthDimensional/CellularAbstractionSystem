import pygame
from typing import Optional, Tuple, Sequence

class Camera:
    """
    Camera class for handling world-to-screen transformations, panning, and zooming in a 2D simulation.
    """

    def __init__(
            self,
            screen_width: int,
            screen_height: int,
            render_buffer: int = 0,
    ) -> None:
        """
        Initializes the Camera.

        :param screen_width:
        :param screen_height:
        :param render_buffer:
        """

        self.x: float = 0
        self.y: float = 0
        self.target_x: float = 0
        self.target_y: float = 0
        self.zoom: float = 1.0
        self.target_zoom: float = 1.0
        self.smoothing: float = 0.15 # lower is smoother
        self.speed: float = 700
        self.zoom_smoothing: float = 0.2 # lower is smoother
        self.is_panning: bool = False
        self.last_mouse_pos: Optional[Sequence[int]] = None
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.render_buffer: int = render_buffer
        self.min_zoom: float = 50.0  # Maximum zoom level
        self.max_zoom: float = 0.01  # Minimum zoom level

    def update(self, keys: Sequence[bool], deltatime: float) -> None:
        """
                Updates the camera position and zoom based on input and time.

                :param keys: Sequence of boolean values representing pressed keys.
                :param deltatime: Time elapsed since last update (in seconds).
                """
        dx = 0
        dy = 0
        if keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_s]:
            dy += 1
        if keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_d]:
            dx += 1

        length = (dx ** 2 + dy ** 2) ** 0.5
        if length > 0:
            dx /= length
            dy /= length

        self.target_x += dx * self.speed * deltatime / self.zoom
        self.target_y += dy * self.speed * deltatime / self.zoom

        if keys[pygame.K_r]:
            self.target_x = 0
            self.target_y = 0

        smoothing_factor = 1 - pow(1 - self.smoothing, deltatime * 60)
        self.x += (self.target_x - self.x) * smoothing_factor
        self.y += (self.target_y - self.y) * smoothing_factor

        threshold = 0.5
        if abs(self.x - self.target_x) < threshold:
            self.x = self.target_x
        if abs(self.y - self.target_y) < threshold:
            self.y = self.target_y

        zoom_smoothing_factor = 1 - pow(1 - self.zoom_smoothing, deltatime * 60)
        self.zoom += (self.target_zoom - self.zoom) * zoom_smoothing_factor

        zoom_threshold = 0.01
        if abs(self.zoom - self.target_zoom) < zoom_threshold:
            self.zoom = self.target_zoom

    def handle_zoom(self, zoom_delta: int) -> None:
        """
        Adjusts the camera zoom level based on mouse wheel input.

        :param zoom_delta: The amount of zoom change (positive for zoom in, negative for zoom out).
        """
        zoom_factor = 1.1
        if zoom_delta > 0:
            self.target_zoom *= zoom_factor
        elif zoom_delta < 0:
            self.target_zoom /= zoom_factor

        self.target_zoom = max(self.max_zoom, min(self.min_zoom, self.target_zoom))

    def start_panning(self, mouse_pos: Sequence[int]) -> None:
        """
        Begins panning the camera.

        :param mouse_pos: The current mouse position as a sequence (x, y).
        """
        self.is_panning = True
        self.last_mouse_pos = mouse_pos

    def stop_panning(self) -> None:
        """
        Stops panning the camera.
        """
        self.is_panning = False
        self.last_mouse_pos = None

    def reset_position(self) -> None:
        """
        Resets the camera position to the origin.
        """
        self.target_x = 0
        self.target_y = 0

    def pan(self, mouse_pos: Sequence[int]) -> None:
        """
        Pans the camera based on mouse movement.

        :param mouse_pos: The current mouse position as a sequence (x, y).
        """
        if self.is_panning and self.last_mouse_pos:
            dx = mouse_pos[0] - self.last_mouse_pos[0]
            dy = mouse_pos[1] - self.last_mouse_pos[1]
            self.x -= dx / self.zoom
            self.y -= dy / self.zoom
            self.target_x = self.x
            self.target_y = self.y
            self.last_mouse_pos = mouse_pos