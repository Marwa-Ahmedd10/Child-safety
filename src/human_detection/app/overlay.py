"""The preview window and what it shows: zones, people with their labels, the counts and
open alerts. Sizes follow the frame's, so 4K and webcam previews look alike."""
import math

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZONE_COLORS = {'hazard': (40, 40, 230), 'exit': (0, 140, 255), 'count': (230, 190, 40), 'ignore': (130, 130, 130),
               'line': (255, 255, 255)}
# Per labeller kind; with whole-body labels, children stand out in yellow.
LABEL_COLORS = {
    'child': {'Child': (0, 215, 255), 'Adult': (255, 180, 70), 'Checking': (200, 200, 200)},
    'age': {'Child': (80, 220, 100), 'Adult': (255, 180, 70), 'Older': (190, 100, 230), 'Unknown': (160, 160, 160)},
}
PERSON_COLOR = (80, 220, 100)  # people without a label
ALERT_COLOR = (40, 40, 220)
WHITE = (255, 255, 255)


def ui_scale(image):
    """1 for a 1280-pixel-wide frame, in proportion for others."""
    return max(image.shape[:2]) / 1280


def text_color(fill):
    """Black on light fills, white on dark ones."""
    blue, green, red = fill
    return (0, 0, 0) if 0.114 * blue + 0.587 * green + 0.299 * red > 150 else WHITE


def draw_tag(image, text, origin, fill, scale):
    """`text` on a filled box whose bottom-left corner is at `origin`, kept inside the image."""
    thickness = max(1, round(2 * scale))
    (width, height), _ = cv2.getTextSize(text, FONT, scale, thickness)
    pad = max(3, height // 3)
    x = int(min(max(origin[0], 0), image.shape[1] - width - 2 * pad))
    y = int(min(max(origin[1], height + 2 * pad), image.shape[0]))
    cv2.rectangle(image, (x, y - height - 2 * pad), (x + width + 2 * pad, y), fill, -1)
    cv2.putText(image, text, (x + pad, y - pad), FONT, scale, text_color(fill), thickness)


def draw_zones(image, zones, counts=None, crossings=None):
    """Shade area zones, outline every zone, and name each at its centre. Count zones
    also show child/adult counts, and counting lines an arrow to their "in" side and,
    when given, their in/out totals."""
    height, width = image.shape[:2]
    ui = ui_scale(image)
    overlay = image.copy()
    outlines = []
    for zone in zones:
        points = np.array([(x * width, y * height) for x, y in zone.polygon], np.int32)
        # Count zones often cover most of the view, so shading them would hide the others.
        if zone.type not in ('count', 'line'):
            cv2.fillPoly(overlay, [points], ZONE_COLORS[zone.type])
        outlines.append((zone, points))
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
    for zone, points in outlines:
        color = ZONE_COLORS[zone.type]
        cv2.polylines(image, [points], True, color, max(2, round(3 * ui)))
        text = zone.name.upper() + (' (ANYONE)' if zone.anyone else ' (ALONE)' if zone.unaccompanied else '')
        if counts and zone.name in counts and zone.type == 'count':
            text += f': {counts[zone.name][0]} CHILDREN, {counts[zone.name][1]} ADULTS'
        if zone.type == 'line':
            (x1, y1), (x2, y2) = points
            middle = ((x1 + x2) // 2, (y1 + y2) // 2)
            length = max(math.hypot(x2 - x1, y2 - y1), 1)
            # The "in" side: the right of the line, from its first point to its second.
            tip = (int(middle[0] - (y2 - y1) / length * 40 * ui), int(middle[1] + (x2 - x1) / length * 40 * ui))
            cv2.arrowedLine(image, middle, tip, color, max(2, round(2 * ui)), tipLength=0.4)
            text = zone.name.upper()
            if crossings and zone.name in crossings:
                text += f': IN {crossings[zone.name]["in"]["total"]} / OUT {crossings[zone.name]["out"]["total"]}'
        scale = 0.75 * ui
        (text_width, _), _ = cv2.getTextSize(text, FONT, scale, max(1, round(2 * scale)))
        x, y = points.mean(axis=0).astype(int)
        draw_tag(image, text, (x - text_width // 2, y), color, scale)


def draw_people(image, people, labels, colors, frame_number, scores=None):
    """Box each person counted, {track_id: (last frame seen, box)}, in their label's colour,
    with a tag naming the label and, from `scores`, how sure it is. People still being
    checked get a thin box and no tag, so a crowd of distant people does not hide the view."""
    ui = ui_scale(image)
    for track_id, (last, (x1, y1, x2, y2)) in people.items():
        label = labels.get(track_id)
        color = colors[label] if label else PERSON_COLOR
        if label == 'Checking':
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            continue
        # A thin box marks someone briefly missed but still counted.
        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, round(2 * ui)) if last == frame_number else 1)
        corner = max(3, round(4 * ui))
        for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            cv2.rectangle(image, (x - corner, y - corner), (x + corner, y + corner), color, -1)
        text = label.upper() if label else 'PERSON'
        score = (scores or {}).get(track_id)
        if score is not None:
            text += f' {score:.2f}'
        draw_tag(image, f'{text}  #{track_id}', (x1 - corner, y1 - corner - 2), color, 0.55 * ui)


def draw_header(image, text):
    """A dark bar across the top with the counts, and a reminder of what they cover."""
    cv2.rectangle(image, (0, 0), (image.shape[1], 80), (25, 25, 25), -1)
    cv2.putText(image, text, (15, 32), FONT, 0.8, (80, 230, 100), 2)
    cv2.putText(image, 'Visible people only. Labels are estimates. Q: quit',
                (15, 62), FONT, 0.5, (240, 240, 240), 1)


def draw_alerts(image, messages):
    """A red banner with a warning sign, centred under the header, for each of the first
    four open alerts."""
    ui = ui_scale(image)
    y = 80 + round(10 * ui)
    for message in messages[:4]:
        text = f'ALERT: {message.upper()}'
        scale, thickness, pad = 0.75 * ui, max(2, round(2 * ui)), round(10 * ui)
        (width, height), _ = cv2.getTextSize(text, FONT, scale, thickness)
        icon = round(height * 1.6)
        # Shrink a long message to fit the frame.
        room = image.shape[1] - icon - 5 * pad
        if width > room:
            scale, width, height = scale * room / width, room, round(height * room / width)
        bottom = y + max(height, icon) + 2 * pad
        left = (image.shape[1] - width - icon - 3 * pad) // 2
        cv2.rectangle(image, (left, y), (left + width + icon + 3 * pad, bottom), ALERT_COLOR, -1)
        middle = left + pad + icon // 2
        triangle = np.array([(middle, bottom - pad - icon), (middle - icon // 2, bottom - pad),
                             (middle + icon // 2, bottom - pad)], np.int32)
        cv2.fillPoly(image, [triangle], WHITE)
        (mark_width, _), _ = cv2.getTextSize('!', FONT, scale, thickness)
        cv2.putText(image, '!', (middle - mark_width // 2, bottom - pad - icon // 8), FONT, scale, ALERT_COLOR,
                    thickness)
        cv2.putText(image, text, (left + 2 * pad + icon, bottom - pad - (max(height, icon) - height) // 2), FONT,
                    scale, WHITE, thickness)
        y = bottom + round(6 * ui)


def offline_view(preview):
    """The last preview under a banner saying the camera is reconnecting."""
    offline = preview.copy()
    cv2.rectangle(offline, (0, 0), (offline.shape[1], 80), ALERT_COLOR, -1)
    cv2.putText(offline, 'CAMERA OFFLINE - reconnecting', (15, 50), FONT, 1.0, WHITE, 2)
    return offline


class Window:
    def __init__(self, title):
        self.title = title
        self.opened = False

    def show(self, image):
        """Show a frame; False once the person quits with Q or Escape or closes the window."""
        if not self.opened:
            # Fit large frames, such as 4K CCTV or portrait video, on screen; the window stays resizable.
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            scale = min(1.0, 1280 / image.shape[1], 800 / image.shape[0])
            cv2.resizeWindow(self.title, int(image.shape[1] * scale), int(image.shape[0] * scale))
            self.opened = True
        cv2.imshow(self.title, image)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            return False
        return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1

    def close(self):
        cv2.destroyAllWindows()
