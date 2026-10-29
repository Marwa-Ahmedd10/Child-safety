"""Zones and counting lines, from a JSON file with points in 0–1 image coordinates (see
zones.example.json): hazard, exit, count and ignore areas, optionally active only at set
hours, and lines that count visitors. A hazard zone can be any area children should not
be in: water, stairs, a kitchen, a car park, machinery."""
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ZONE_TYPES = ('hazard', 'exit', 'count', 'ignore', 'line')
DEFAULTS = {'confirm_frames': 10, 'release_frames': 30, 'accompany_distance': 1.5}


@dataclass
class Zone:
    name: str
    type: str
    polygon: list
    max_children: int = None
    max_children_per_adult: float = None
    anyone: bool = False  # a hazard zone that alerts on any person, not only possible children
    hours: list = None    # local times the zone applies, e.g. ["20:00-07:00"]; always if absent
    unaccompanied: bool = False  # a hazard zone that alerts only on children with no adult nearby
    after_seconds: float = 0     # how long one person must stay in a hazard or exit zone before it alerts
    alert: str = None            # the alert's text, e.g. "Child near the stairs"; by default it names the zone

    def contains(self, point):
        x, y = point
        inside = False
        for (x1, y1), (x2, y2) in zip(self.polygon, self.polygon[1:] + self.polygon[:1], strict=True):
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
        return inside

    def active(self, now):
        """Whether the zone applies at `now`: always, or within one of its hours."""
        if not self.hours:
            return True
        minute = now.hour * 60 + now.minute
        # A span whose end is before its start runs overnight.
        return any(start <= minute < end if start < end else minute >= start or minute < end
                   for start, end in self.hours)

    def needs_labels(self):
        """Whether the zone's rule depends on child/adult labels."""
        return self.type in ('exit', 'count') or (self.type == 'hazard' and not self.anyone)


def parse_zone(raw):
    try:
        zone = Zone(**raw)
    except TypeError as exc:
        raise ValueError(f'Zone "{raw.get("name", "?")}": {exc}') from None
    if zone.type not in ZONE_TYPES:
        raise ValueError(f'Zone "{zone.name}": type must be one of {", ".join(ZONE_TYPES)}')
    # A counting line has two points: from its first to its second, "in" is to the right.
    enough = len(zone.polygon) == 2 if zone.type == 'line' else len(zone.polygon) >= 3
    if not enough or not all(len(p) == 2 and 0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in zone.polygon):
        need = 'exactly 2' if zone.type == 'line' else '3 or more'
        raise ValueError(f'Zone "{zone.name}": polygon needs {need} [x, y] points between 0 and 1')
    zone.polygon = [tuple(map(float, p)) for p in zone.polygon]
    if zone.type == 'line' and math.dist(*zone.polygon) < 0.01:
        raise ValueError(f'Zone "{zone.name}": the line\'s two points are the same; draw it again')
    if (zone.anyone or zone.unaccompanied) and zone.type != 'hazard':
        option = 'anyone' if zone.anyone else 'unaccompanied'
        raise ValueError(f'Zone "{zone.name}": only hazard zones take "{option}"')
    if zone.anyone and zone.unaccompanied:
        raise ValueError(f'Zone "{zone.name}": "anyone" and "unaccompanied" cannot both be set')
    if (zone.after_seconds or zone.alert is not None) and zone.type not in ('hazard', 'exit'):
        raise ValueError(f'Zone "{zone.name}": only hazard and exit zones take "after_seconds" and "alert"')
    if isinstance(zone.after_seconds, bool) or not isinstance(zone.after_seconds, (int, float)) \
            or zone.after_seconds < 0:
        raise ValueError(f'Zone "{zone.name}": "after_seconds" must be a number of seconds, 0 or more')
    if zone.alert is not None and (not isinstance(zone.alert, str) or not zone.alert.strip()):
        raise ValueError(f'Zone "{zone.name}": "alert" must be the text to show, or left out')
    if zone.hours is not None:
        zone.hours = [parse_span(zone.name, span) for span in zone.hours]
    return zone


def parse_span(name, span):
    """'20:00-07:00' as (start, end) minutes after midnight."""
    try:
        start, end = (datetime.strptime(part.strip(), '%H:%M') for part in span.split('-'))
    except (ValueError, AttributeError):
        raise ValueError(f'Zone "{name}": hours must look like "20:00-07:00", not {span!r}') from None
    return start.hour * 60 + start.minute, end.hour * 60 + end.minute


def load_zones(path):
    """Return (zones, settings) from a zones JSON file."""
    config = json.loads(Path(path).read_text())
    settings = {key: config.get(key, value) for key, value in DEFAULTS.items()}
    return [parse_zone(raw) for raw in config['zones']], settings


def foot(box, size):
    """Where a person stands, as 0–1 image coordinates: the bottom centre of their box.
    Clamped just inside the image so zones drawn to the edge still contain it."""
    x1, y1, x2, y2 = box
    width, height = size
    return (min(max((x1 + x2) / 2 / width, 0.001), 0.999), min(max(y2 / height, 0.001), 0.999))
