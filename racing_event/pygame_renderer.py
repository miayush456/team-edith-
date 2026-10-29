"""Read-only Pygame presentation for validated SYNAPSE replays.

Pygame consumes the Python-authored replay exchange. It never steps physics,
infers contacts, or calculates results. Only position and heading interpolate.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .replay import ReplayClock, sample_replay
from .replay_group import ReplayGroup, load_replay_group
from .trajectory import load_replay


MIN_WINDOW_SIZE = (640, 480)
MIN_SPEED = 0.125
MAX_SPEED = 16.0
BACKGROUND = (5, 8, 14)
PANEL = (15, 24, 36)
TEXT = (232, 244, 255)
MUTED = (158, 180, 202)
ACCENT = (69, 243, 255)
WARNING = (255, 113, 148)
GOLD = (255, 209, 102)
EVENT_LABELS = {
    "off_track_start": "Off track",
    "off_track_end": "Back on track",
    "barrier_contact": "Barrier contact",
    "lap_complete": "Lap complete",
    "checkpoint": "Progress zone reached",
}
RESULT_LABELS = {
    "finished": "FINISHED",
    "dnf_crash": "DNF - barrier contact",
    "dnf_stuck": "DNF - stuck",
    "dnf_backward": "DNF - backward",
    "dnf_timeout": "DNF - timeout",
}
RESULT_COMPACT = {
    "finished": "FINISHED", "dnf_crash": "DNF crash", "dnf_stuck": "DNF stuck",
    "dnf_backward": "DNF backward", "dnf_timeout": "DNF timeout",
}


class PygameReplayRenderer:
    """Present one replay or a validated group on one visual clock.

    Group telemetry/events always belong to the selected ghost, even if hidden.
    Visibility and path controls are presentation preferences, never data edits.
    """

    def __init__(self, data: Mapping[str, Any] | ReplayGroup, surface: Any, clock: ReplayClock | None = None):
        import pygame

        self.group = data if isinstance(data, ReplayGroup) else None
        self.recordings = tuple(p.data for p in data.participants) if self.group else (data,)
        self.selected_index = 0
        self.visible = [True] * len(self.recordings)
        self.show_paths = [True] * len(self.recordings)
        self.data = self.recordings[0]
        self.track = self.data["track"]
        final_tick = max(d["result"]["steps"] for d in self.recordings)
        self.clock = clock if clock is not None else ReplayClock(self.data["tick_hz"], final_tick)
        if self.clock.tick_hz != self.data["tick_hz"] or self.clock.final_tick != final_tick:
            raise ValueError("visual clock must match the recording/group tick range")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 19)
        # Scan bounded recording data once, never during animation or a seek.
        self._events = tuple(tuple(f["tick"] for f in d["frames"] if f["events"]) for d in self.recordings)
        self.event_ticks = self._events[0]
        self._bounds = self._world_bounds()
        self.show_help = False
        self.dragging = False
        self.hover_event: int | None = None
        self.resize(surface)

    def _world_bounds(self) -> tuple[float, float, float, float]:
        points = [point for name in ("left_boundary", "right_boundary", "barrier_left", "barrier_right", "centerline")
                  for point in self.track.get(name, [])]
        if not points:
            raise ValueError("replay track has no drawable geometry")
        min_x = min(float(point[0]) for point in points)
        max_x = max(float(point[0]) for point in points)
        min_y = min(float(point[1]) for point in points)
        max_y = max(float(point[1]) for point in points)
        # Include the whole recorded path and a conservative footprint radius.
        # This also contains every intermediate shortest-angle pose, without
        # inflating the actual car drawn on screen.
        for data in self.recordings:
            radius = math.hypot(data["car"]["length_m"], data["car"]["width_m"]) / 2
            for frame in data["frames"]:
                min_x, max_x = min(min_x, frame["x"] - radius), max(max_x, frame["x"] + radius)
                min_y, max_y = min(min_y, frame["y"] - radius), max(max_y, frame["y"] + radius)
        return min_x, max_x, min_y, max_y

    def resize(self, surface: Any) -> None:
        """Rebuild the static layer only when the surface size changes."""
        import pygame

        self.surface = surface
        size = surface.get_size()
        if getattr(self, "_cached_size", None) == size:
            return
        self.width, self.height = size
        if self.width < MIN_WINDOW_SIZE[0] or self.height < MIN_WINDOW_SIZE[1]:
            raise ValueError("viewer size must be at least 640x480")
        self._cached_size = size
        header_height = max(80, 60 + len(self.recordings) * 18) if self.group else 80
        self.header_rect = pygame.Rect(0, 0, self.width, header_height)
        self.footer_rect = pygame.Rect(0, self.height - 166, self.width, 166)
        panel_width = 216 if self.width < 960 else 272
        content_height = self.footer_rect.top - self.header_rect.bottom - 24
        self.panel_rect = pygame.Rect(self.width - panel_width - 12, header_height + 12, panel_width, content_height)
        self.track_rect = pygame.Rect(12, header_height + 12, self.panel_rect.left - 24, content_height)
        self.scrubber_rect = pygame.Rect(20, self.footer_rect.top + 27, self.width - 40, 32)
        self.buttons: dict[str, Any] = {}
        x = 16
        for action, width in (("play", 78), ("restart", 76), ("previous_event", 94), ("next_event", 94),
                              ("slower", 36), ("speed", 58), ("faster", 36), ("help", 70)):
            self.buttons[action] = pygame.Rect(x, self.footer_rect.top + 86, width, 34)
            x += width + 8
        self._origin_x, self._origin_y, self._scale = self._fit_track()
        self.paths = tuple([self._screen_point(float(f["x"]), float(f["y"])) for f in d["frames"]]
                           for d in self.recordings)
        self.path = self.paths[self.selected_index]
        self._background = pygame.Surface(size)
        self._background.fill(BACKGROUND)
        self._background.set_clip(self.track_rect)
        self._draw_track(pygame, self._background)
        self._path_layers = []
        if self.group:
            for index, path in enumerate(self.paths):
                layer = pygame.Surface(size, pygame.SRCALPHA)
                layer.set_clip(self.track_rect)
                # Different dash cadences plus numbered legends distinguish
                # paths without relying only on palette differences.
                stride = index + 1
                for start in range(0, len(path) - 1, stride * 3):
                    segment = path[start:start + stride * 2 + 1]
                    if len(segment) >= 2:
                        pygame.draw.lines(layer, self.group.participants[index].color, False, segment, 1)
                self._path_layers.append(layer)
        elif len(self.path) >= 2:
            pygame.draw.lines(self._background, GOLD, False, self.path, 1)
        self._background.set_clip(None)
        self._refresh_markers(pygame)
        self.dragging = False
        self.hover_event = None

    def _refresh_markers(self, pygame: Any) -> None:
        # Symbols preserve event type without relying on color. Multiple events
        # on one tick share a seek target and remain listed in the readout.
        self.marker_rects = {
            tick: pygame.Rect(self._tick_x(tick) - 6, self.scrubber_rect.top + 16, 13, 16)
            for tick in self.event_ticks
        }
        # Endpoint symbols extend beyond the line's first/last pixel. Clear
        # their full footprint when switching ghosts, not just the rail width.
        self._background.fill(BACKGROUND, self.scrubber_rect.inflate(16, 0))
        self._draw_timeline_background(pygame)

    def select_ghost(self, index: int) -> None:
        """Select exact telemetry/events without moving or restarting the clock."""
        import pygame

        if not 0 <= index < len(self.recordings):
            raise ValueError("selected ghost is outside the group")
        self.selected_index = index
        self.data = self.recordings[index]
        self.path = self.paths[index]
        self.event_ticks = self._events[index]
        self.hover_event = None
        self.dragging = False
        self._refresh_markers(pygame)

    def _fit_track(self) -> tuple[float, float, float]:
        min_x, max_x, min_y, max_y = self._bounds
        area = self.track_rect.inflate(-28, -28)
        scale = min(area.width / max(max_x - min_x, 1.0), area.height / max(max_y - min_y, 1.0))
        return (area.centerx - (min_x + max_x) * scale / 2,
                area.centery + (min_y + max_y) * scale / 2, scale)

    def _screen_point(self, x: float, y: float) -> tuple[int, int]:
        return round(self._origin_x + x * self._scale), round(self._origin_y - y * self._scale)

    def _polyline(self, points: Any) -> list[tuple[int, int]]:
        return [self._screen_point(float(x), float(y)) for x, y in points]

    def _draw_track(self, pygame: Any, target: Any) -> None:
        colors = {"left_boundary": (62, 226, 210), "right_boundary": (62, 226, 210),
                  "barrier_left": (255, 91, 132), "barrier_right": (255, 91, 132), "centerline": (76, 92, 112)}
        for name in ("barrier_left", "barrier_right", "left_boundary", "right_boundary", "centerline"):
            points = self._polyline(self.track.get(name, []))
            if len(points) >= 2:
                pygame.draw.lines(target, colors[name], True, points, 2 if "barrier" in name else 1)

    def _tick_x(self, tick: float) -> int:
        return round(self.scrubber_rect.left + tick / self.clock.final_tick * (self.scrubber_rect.width - 1))

    def _draw_timeline_background(self, pygame: Any) -> None:
        y = self.scrubber_rect.top + 8
        pygame.draw.line(self._background, MUTED, (self.scrubber_rect.left, y), (self.scrubber_rect.right - 1, y), 3)
        for tick in self.event_ticks:
            x = self._tick_x(tick)
            events = self.data["frames"][tick]["events"]
            y = self.scrubber_rect.top + 23
            if "barrier_contact" in events:
                pygame.draw.line(self._background, WARNING, (x - 4, y - 4), (x + 4, y + 4), 2)
                pygame.draw.line(self._background, WARNING, (x + 4, y - 4), (x - 4, y + 4), 2)
            elif "lap_complete" in events:
                pygame.draw.polygon(self._background, GOLD, [(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)])
            elif "off_track_start" in events:
                pygame.draw.polygon(self._background, WARNING, [(x - 4, y - 4), (x + 4, y - 4), (x, y + 4)])
            else:
                pygame.draw.polygon(self._background, ACCENT, [(x - 4, y + 4), (x + 4, y + 4), (x, y - 4)])

    def _draw_car(self, pygame: Any, pose: Mapping[str, float], off_track: bool, color: Any = None) -> None:
        x, y, heading = float(pose["x"]), float(pose["y"]), float(pose["heading"])
        half_length = float(self.data["car"]["length_m"]) / 2
        half_width = float(self.data["car"]["width_m"]) / 2
        forward = (math.cos(heading), math.sin(heading))
        lateral = (-math.sin(heading), math.cos(heading))

        def point(longitudinal: float, transverse: float) -> tuple[int, int]:
            return self._screen_point(x + longitudinal * forward[0] + transverse * lateral[0],
                                      y + longitudinal * forward[1] + transverse * lateral[1])

        corners = [point(half_length, half_width), point(half_length, -half_width),
                   point(-half_length, -half_width), point(-half_length, half_width)]
        # No minimum-sized sprite, wheels, shadow or nose outside the recorded
        # metric rectangle: the displayed footprint remains geometry truth.
        pygame.draw.polygon(self.surface, color if color is not None else WARNING if off_track else ACCENT, corners)
        pygame.draw.line(self.surface, BACKGROUND, point(half_length * 0.35, half_width * 0.65),
                         point(half_length * 0.35, -half_width * 0.65), 1)

    def _text(self, value: str, rect: Any, *, small: bool = True, color: Any = TEXT, centered: bool = False) -> None:
        font = self.small_font if small else self.font
        value = "".join(c if c.isprintable() else " " for c in value)
        # Untrusted participant/track names can be long; fit them to their own
        # region rather than letting text escape over the track or controls.
        if font.size(value)[0] > rect.width:
            while value and font.size(value + "...")[0] > rect.width:
                value = value[:-1]
            value += "..."
        image = font.render(value, True, color)
        destination = image.get_rect(center=rect.center) if centered else image.get_rect(topleft=rect.topleft)
        self.surface.blit(image, destination)

    def _seek(self, tick: int) -> None:
        self.clock.pause()
        self.clock.seek_tick(tick)

    def _event_target(self, direction: int) -> int | None:
        position = self.clock.tick_position
        index = bisect_left(self.event_ticks, position) - 1 if direction < 0 else bisect_right(self.event_ticks, position)
        return self.event_ticks[index] if 0 <= index < len(self.event_ticks) else None

    def _enabled(self, action: str) -> bool:
        if action in ("previous_event", "next_event"):
            return self._event_target(-1 if action == "previous_event" else 1) is not None
        if action == "slower":
            return self.clock.speed > MIN_SPEED
        if action == "faster":
            return self.clock.speed < MAX_SPEED
        return action != "speed"

    def _activate(self, action: str) -> None:
        if not self._enabled(action):
            return
        if action == "play":
            if self.clock.tick_position >= self.clock.final_tick:
                self.clock.restart()
            else:
                self.clock.toggle()
        elif action == "restart":
            self.clock.restart()
        elif action in ("previous_event", "next_event"):
            target = self._event_target(-1 if action == "previous_event" else 1)
            if target is not None:
                self._seek(target)
        elif action in ("slower", "faster"):
            self.clock.set_speed(max(MIN_SPEED, min(MAX_SPEED, self.clock.speed * (0.5 if action == "slower" else 2))))
        elif action == "help":
            self.show_help = not self.show_help

    def _marker_at(self, position: tuple[int, int]) -> int | None:
        # Nearest marker wins when dense recordings put several on one pixel;
        # previous/next event still reaches every recorded event tick.
        candidates = (tick for tick, rect in self.marker_rects.items() if rect.collidepoint(position))
        return min(candidates, key=lambda tick: abs(self._tick_x(tick) - position[0]), default=None)

    def _scrub(self, x: int) -> None:
        fraction = (x - self.scrubber_rect.left) / (self.scrubber_rect.width - 1)
        self._seek(round(max(0.0, min(1.0, fraction)) * self.clock.final_tick))

    def handle_event(self, event: Any) -> bool:
        """Consume real Pygame input. Return False when the viewer should close."""
        import pygame

        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            size = (max(MIN_WINDOW_SIZE[0], event.w), max(MIN_WINDOW_SIZE[1], event.h))
            self.resize(pygame.display.set_mode(size, pygame.RESIZABLE))
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            actions = {pygame.K_SPACE: "play", pygame.K_r: "restart", pygame.K_p: "previous_event", pygame.K_n: "next_event",
                       pygame.K_LEFTBRACKET: "slower", pygame.K_MINUS: "slower", pygame.K_KP_MINUS: "slower",
                       pygame.K_RIGHTBRACKET: "faster", pygame.K_EQUALS: "faster", pygame.K_PLUS: "faster",
                       pygame.K_KP_PLUS: "faster", pygame.K_h: "help", pygame.K_SLASH: "help", pygame.K_QUESTION: "help"}
            if key in actions:
                self._activate(actions[key])
            elif key in (pygame.K_LEFT, pygame.K_RIGHT):
                increment = self.clock.tick_hz if getattr(event, "mod", 0) & pygame.KMOD_SHIFT else 1
                self._seek(math.floor(self.clock.tick_position) + (-increment if key == pygame.K_LEFT else increment))
            elif key in (pygame.K_HOME, pygame.K_END):
                self._seek(0 if key == pygame.K_HOME else self.clock.final_tick)
            elif self.group:
                if key == pygame.K_TAB:
                    direction = -1 if getattr(event, "mod", 0) & pygame.KMOD_SHIFT else 1
                    self.select_ghost((self.selected_index + direction) % len(self.recordings))
                elif pygame.K_1 <= key <= pygame.K_4 and key - pygame.K_1 < len(self.recordings):
                    self.select_ghost(key - pygame.K_1)
                elif key == pygame.K_v:
                    self.visible[self.selected_index] = not self.visible[self.selected_index]
                elif key == pygame.K_t:
                    self.show_paths[self.selected_index] = not self.show_paths[self.selected_index]
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            marker = self._marker_at(event.pos)
            if marker is not None:
                self._seek(marker)
            elif self.scrubber_rect.collidepoint(event.pos):
                self.dragging = True
                self._scrub(event.pos[0])
            else:
                for action, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        self._activate(action)
                        break
        elif event.type == pygame.MOUSEMOTION:
            self.hover_event = self._marker_at(event.pos)
            if self.dragging:
                self._scrub(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self._scrub(event.pos[0])
            self.dragging = False
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.dragging = False
        return True

    def draw(self, sampled: Mapping[str, Any] | None = None) -> None:
        import pygame

        self.resize(self.surface)
        sampled = sample_replay(self.data, self.clock) if sampled is None else sampled
        frame = sampled["frame"]
        result = self.data["result"]
        self.surface.blit(self._background, (0, 0))
        old_clip = self.surface.get_clip()
        self.surface.set_clip(self.track_rect)
        if self.group:
            for index, layer in enumerate(self._path_layers):
                if self.visible[index] and self.show_paths[index]:
                    self.surface.blit(layer, (0, 0))
            # Selected ghost draws last when standing starts coincide. Numbers
            # are labels offset from the metric car, not oversized car sprites.
            order = [i for i in range(len(self.recordings)) if i != self.selected_index] + [self.selected_index]
            for index in order:
                if not self.visible[index]:
                    continue
                ghost = sample_replay(self.recordings[index], self.clock)
                color = self.group.participants[index].color
                self._draw_car(pygame, ghost["pose"], ghost["frame"]["off_track"], color)
                x, y = self._screen_point(ghost["pose"]["x"], ghost["pose"]["y"])
                self._text(str(index + 1), pygame.Rect(x + 6, y + 4 + index * 13, 16, 16), color=color)
        else:
            self._draw_car(pygame, sampled["pose"], bool(frame["off_track"]))
        self.surface.set_clip(old_clip)
        pygame.draw.rect(self.surface, MUTED, self.track_rect, 1)
        pygame.draw.rect(self.surface, PANEL, self.panel_rect, border_radius=6)

        self._text(f"SYNAPSE / {self.track['meta']['display_name']}", pygame.Rect(16, 9, self.width - 32, 25), small=False)
        self._text("GROUP | Shared standing tick 0 | Hold final pose | Results unranked" if self.group else
                   f"{self.data['source']['participant_id']}  |  {self.data['source']['mode']}  |  Read-only replay",
                   pygame.Rect(16, 35, self.width - 32, 20), color=MUTED)
        timing = f"lap {result['lap_time_s']:.3f} s" if result["lap_time_s"] is not None else f"elapsed {result['elapsed_s']:.3f} s; no lap"
        if self.group:
            for index, participant in enumerate(self.group.participants):
                final = participant.data["result"]
                duration = f"lap {final['lap_time_s']:.3f} s" if final["lap_time_s"] is not None else f"elapsed {final['elapsed_s']:.3f} s"
                state = "HOLD FINAL" if self.clock.tick_position > final["steps"] else "AT END" if self.clock.tick_position == final["steps"] else "IN PROGRESS"
                visibility = "ON" if self.visible[index] else "HIDDEN"
                path = "path" if self.show_paths[index] else "no path"
                # Reserve outcome/time/hold space independently of untrusted
                # labels, so a long name cannot hide a DNF or final-pose hold.
                summary = f"final: {RESULT_COMPACT[final['status']]} | {duration} | {state}"
                summary_width = self.small_font.size(summary)[0]
                result_x = max(16 + self.width // 3, self.width - 16 - summary_width)
                self._text(f"{'>' if index == self.selected_index else ' '} {index + 1} [{visibility}/{path}] {participant.label}",
                           pygame.Rect(16, 57 + index * 18, result_x - 24, 18), color=participant.color)
                self._text(summary, pygame.Rect(result_x, 57 + index * 18, self.width - result_x - 16, 18), color=participant.color)
        else:
            self._text(f"Recorded final: {RESULT_LABELS[result['status']]} | {timing} | at tick {result['steps']}",
                       pygame.Rect(16, 57, self.width - 32, 20), color=GOLD)

        current_status = RESULT_LABELS[result["status"]] if frame["tick"] == result["steps"] else "IN PROGRESS"
        if self.show_help and not self.group:
            lines = ["KEYBOARD / ? to close", "Space: play / pause", "R: restart and play", "Left / Right: one tick",
                     "Shift + Left / Right: 1 s", "Home / End: endpoints", "[ / ] or - / +: speed",
                     "P / N: prev / next event", "H / ?: keyboard help", "Esc / Q: close viewer", "Seeking pauses playback."]
            for index, line in enumerate(lines):
                self._text(line, pygame.Rect(self.panel_rect.x + 10, self.panel_rect.y + 10 + index * 18,
                                            self.panel_rect.width - 20, 18), color=TEXT if index == 0 else MUTED)
        elif not self.show_help:
            lines = [f"TICK {frame['tick']} / {result['steps']}",
                     f"Time {frame['tick'] / self.data['tick_hz']:.3f} / {result['elapsed_s']:.3f} s",
                     "OFF TRACK" if frame["off_track"] else "ON TRACK", current_status,
                     f"Speed    {frame['speed']:.2f} m/s", f"Lateral  {frame['lateral_velocity']:.2f} m/s",
                     f"Yaw       {frame['yaw_rate']:.3f} rad/s", f"Steer     {frame['steer']:.3f} rad",
                     f"Command {frame['steering_command']:+.3f}", f"Drive       {frame['throttle_brake']:+.3f}"]
            spacing = min(27, (self.panel_rect.height - (12 if self.group else 20)) // len(lines))
            if self.group:
                lines[0] = f"GHOST {self.selected_index + 1} | TICK {frame['tick']}/{result['steps']}"
                if frame["tick"] == result["steps"]:
                    lines[3] = RESULT_COMPACT[result["status"]]
                if self.clock.tick_position > result["steps"]:
                    lines[3] = "HOLD FINAL: " + RESULT_COMPACT[result["status"]]
            for index, line in enumerate(lines):
                color = WARNING if index == 2 and frame["off_track"] else TEXT
                if index == 3 and frame["tick"] == result["steps"]:
                    color = GOLD if result["status"] == "finished" else WARNING
                self._text(line, pygame.Rect(self.panel_rect.x + 10, self.panel_rect.y + (7 if self.group else 10) + index * spacing,
                                             self.panel_rect.width - 20, spacing),
                           small=spacing < 18 or index not in (0, 2, 3), color=color)

        if self.show_help and self.group:
            # Four ghosts at the minimum size leave a short side panel. Use a
            # two-column help sheet in the body, so every command stays visible.
            area = self.track_rect.union(self.panel_rect)
            pygame.draw.rect(self.surface, PANEL, area)
            lines = ["GROUP KEYS / H or ? to close", "Space: play / pause", "R: restart and play", "Left / Right: one tick",
                     "Shift + Left / Right: 1 s", "Home / End: group endpoints", "[ / ] or - / +: speed", "P / N: selected ghost events",
                     "1-4: select ghost telemetry", "Tab / Shift+Tab: cycle ghost", "V: selected ghost on / hidden", "T: selected path on / off",
                     "Short ghosts HOLD FINAL", "No interpolation past end", "Seeking pauses playback", "Esc / Q: close viewer"]
            for index, line in enumerate(lines):
                column, row = divmod(index, 8)
                self._text(line, pygame.Rect(area.x + 10 + column * area.width // 2, area.y + 7 + row * 18,
                                             area.width // 2 - 20, 18), color=TEXT if index == 0 else MUTED)

        playback = "Playing" if self.clock.playing else "End of recording" if self.clock.tick_position >= self.clock.final_tick else "Paused"
        position = self.clock.tick_position if self.group else sampled["tick_position"]
        self._text(f"{playback} | Shared tick {math.floor(position)} | {self.clock.seconds:.3f} / {self.clock.duration_seconds:.3f} s" if self.group else
                   f"{playback} | Tick {frame['tick']} | {frame['tick'] / self.data['tick_hz']:.3f} / {result['elapsed_s']:.3f} s",
                   pygame.Rect(20, self.footer_rect.top + 4, self.width - 40, 20), color=MUTED)
        pygame.draw.circle(self.surface, ACCENT, (self._tick_x(position), self.scrubber_rect.top + 8), 6)
        event_tick = frame["tick"] if self.hover_event is None else self.hover_event
        events = self.data["frames"][event_tick]["events"]
        event_text = ", ".join(EVENT_LABELS[event] for event in events) or ("none" if self.event_ticks else "none in this recording")
        self._text((f"Ghost {self.selected_index + 1} " if self.group else "") +
                   f"{'Marker' if self.hover_event is not None else 'Events'} @ tick {event_tick}: {event_text}",
                   pygame.Rect(20, self.footer_rect.top + 63, self.width - 40, 20))
        labels = {"play": "Pause" if self.clock.playing else "Play", "restart": "Restart", "previous_event": "Prev event",
                  "next_event": "Next event", "slower": "-", "faster": "+", "speed": f"{self.clock.speed:g}x", "help": "Keys ?"}
        for action, rect in self.buttons.items():
            enabled = self._enabled(action)
            if action != "speed":
                pygame.draw.rect(self.surface, (31, 49, 67) if enabled else PANEL, rect, border_radius=4)
                pygame.draw.rect(self.surface, ACCENT if enabled else (54, 67, 80), rect, 1, border_radius=4)
            self._text(labels[action], rect.inflate(-4, 0), centered=True, color=TEXT if enabled or action == "speed" else MUTED)
        self._text("1-4 / Tab: select | V: ghost on/off | T: path on/off | P/N: events | ?: all keys" if self.group else
                   "Space: play | Arrows: tick | Shift: 1 s | Home/End | [ ]: speed | P/N: events | ?: help",
                   pygame.Rect(16, self.footer_rect.top + 134, self.width - 32, 20), color=MUTED)


def run_pygame(
    replay_path: str | Path | None = None,
    *,
    group_path: str | Path | None = None,
    size: tuple[int, int] = (1280, 800),
    speed: float = 1.0,
    headless: bool = False,
    start_seconds: float = 0.0,
    max_frames: int | None = None,
    screenshot: str | Path | None = None,
    start_tick: int | None = None,
    seek_event: str | None = None,
    show_help: bool = False,
    selected_ghost: int = 1,
) -> None:
    """Run a resizable viewer, or save one deterministic paused headless frame."""
    if (replay_path is None) == (group_path is None):
        raise ValueError("provide exactly one replay or group")
    source_path = Path(group_path if group_path is not None else replay_path)
    if screenshot is not None:
        screenshot = Path(screenshot)
        if screenshot.suffix.lower() != ".png" or screenshot.resolve() == source_path.resolve():
            raise ValueError("screenshot must be a .png distinct from the recording")
        headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    data = load_replay_group(source_path) if group_path is not None else load_replay(source_path)
    if screenshot is not None:
        sources = [source_path]
        if isinstance(data, ReplayGroup):
            sources += [source_path.parent / p["file"] for p in data.manifest["participants"]]
        if any(screenshot.resolve() == p.resolve() or screenshot.exists() and os.path.samefile(screenshot, p)
               for p in sources):
            raise ValueError("screenshot must be distinct from the recording and all group inputs")
    tick_hz = data.tick_hz if isinstance(data, ReplayGroup) else data["tick_hz"]
    final_tick = data.final_tick if isinstance(data, ReplayGroup) else data["result"]["steps"]
    clock = ReplayClock(tick_hz, final_tick, seconds=start_seconds, speed=speed, playing=not headless)
    if start_tick is not None:
        clock.seek_tick(start_tick)
    pygame.init()
    try:
        surface = pygame.display.set_mode(size, pygame.RESIZABLE)
        renderer = PygameReplayRenderer(data, surface, clock)
        renderer.select_ghost(selected_ghost - 1)
        renderer.show_help = show_help
        if seek_event is not None:
            if seek_event not in ("first", "last", "previous", "next"):
                raise ValueError("event seek must be first, last, previous or next")
            target = ((renderer.event_ticks[0 if seek_event == "first" else -1] if renderer.event_ticks else None)
                      if seek_event in ("first", "last") else renderer._event_target(-1 if seek_event == "previous" else 1))
            if target is None:
                raise ValueError("no recorded event in requested direction for selected ghost")
            renderer._seek(target)
        if headless:
            renderer.draw()
            pygame.display.flip()
            if screenshot is not None:
                pygame.image.save(renderer.surface, str(screenshot))
            return
        frame_count = 0
        running = True
        wall_clock = pygame.time.Clock()
        while running:
            # Advance before input so seeks/restart render their exact target,
            # without consuming a stale wall-time delta after the user's action.
            clock.advance(wall_clock.tick(60) / 1000.0)
            for event in pygame.event.get():
                if not renderer.handle_event(event):
                    running = False
                    break
            if not running:
                break
            renderer.draw()
            pygame.display.flip()
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        pygame.quit()


__all__ = ["PygameReplayRenderer", "run_pygame"]
