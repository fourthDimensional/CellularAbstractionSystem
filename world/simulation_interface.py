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

    def get_real_coordinates(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """
        Converts screen coordinates to world coordinates.

        :param screen_x: X coordinate on the screen.
        :param screen_y: Y coordinate on the screen.
        :return: Tuple of (world_x, world_y).
        """
        world_x = (screen_x - self.screen_width // 2 + self.x * self.zoom) / self.zoom
        world_y = (screen_y - self.screen_height // 2 + self.y * self.zoom) / self.zoom
        return world_x, world_y

    def is_in_view(self, obj_x: float, obj_y: float, margin: float = 0) -> bool:
        """
        Checks if an object is within the camera's view.

        :param obj_x: X coordinate in world space.
        :param obj_y: Y coordinate in world space.
        :param margin: Additional margin to expand the view area.
        :return: True if the object is in view, False otherwise.
        """
        half_w = (self.screen_width + (self.render_buffer * self.zoom)) / (2 * self.zoom)
        half_h = (self.screen_height + (self.render_buffer * self.zoom)) / (2 * self.zoom)
        cam_left = self.x - half_w
        cam_right = self.x + half_w
        cam_top = self.y - half_h
        cam_bottom = self.y + half_h
        return (
                cam_left - margin <= obj_x <= cam_right + margin
                and cam_top - margin <= obj_y <= cam_bottom + margin
        )

    def world_to_screen(self, obj_x: float, obj_y: float) -> Tuple[int, int]:
        """
        Converts world coordinates to screen coordinates.

        :param obj_x: X coordinate in world space.
        :param obj_y: Y coordinate in world space.
        :return: Tuple of (screen_x, screen_y) in pixels.
        """
        screen_x = (obj_x - self.x) * self.zoom + self.screen_width // 2
        screen_y = (obj_y - self.y) * self.zoom + self.screen_height // 2
        return int(screen_x), int(screen_y)

    def get_relative_size(self, world_size: float) -> int:
        """
        Converts a world size (e.g., radius or width/height) to screen pixels.

        :param world_size: Size in world units.
        :return: Size in screen pixels.
        """
        return int(world_size * self.zoom)