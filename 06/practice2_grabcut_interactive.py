import cv2
import numpy as np
import os

IMAGE_PATH = "../03/data/Lena.png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

mask      = np.zeros((h, w), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

rect      = [0, 0, 0, 0]
drawing   = False
rect_done = False
save_count = 0

DRAW_FG   = {"color": (0, 255, 0), "val": cv2.GC_FGD}
DRAW_BG   = {"color": (0, 0, 255), "val": cv2.GC_BGD}
current   = DRAW_FG

HELP = [
    "LMB drag : rect (init) / foreground",
    "RMB drag : background",
    "ENTER    : run GrabCut",
    "r        : reset",
    "s        : save result",
    "q / ESC  : quit",
]


def apply_grabcut():
    global mask, bgd_model, fgd_model
    if not rect_done:
        return
    x, y, x2, y2 = rect
    r = (min(x, x2), min(y, y2), abs(x2 - x), abs(y2 - y))
    cv2.grabCut(img, mask, r, bgd_model, fgd_model, 5,
                cv2.GC_INIT_WITH_RECT if np.all(mask == 0) else cv2.GC_INIT_WITH_MASK)


def get_result():
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.bitwise_and(img, img, mask=fg)


def draw_overlay(base):
    out = base.copy()
    overlay = img.copy()
    overlay[mask == cv2.GC_BGD]    = (0, 0, 80)
    overlay[mask == cv2.GC_PR_BGD] = (0, 0, 40)
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    if rect_done:
        x, y, x2, y2 = rect
        cv2.rectangle(out, (min(x,x2), min(y,y2)), (max(x,x2), max(y,y2)), (255,255,0), 1)
    for i, line in enumerate(HELP):
        cv2.putText(out, line, (8, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    return out


def mouse_cb(event, x, y, flags, _):
    global drawing, rect, rect_done, mask, bgd_model, fgd_model, current

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if not rect_done:
            rect = [x, y, x, y]
        else:
            current = DRAW_FG
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        current = DRAW_BG

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if not rect_done:
            rect[2], rect[3] = x, y
        else:
            cv2.circle(mask, (x, y), 5, current["val"], -1)

    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        drawing = False
        if not rect_done:
            rect[2], rect[3] = x, y
            rect_done = True


def reset():
    global mask, bgd_model, fgd_model, rect, rect_done, drawing
    mask[:]      = 0
    bgd_model[:] = 0
    fgd_model[:] = 0
    rect         = [0, 0, 0, 0]
    rect_done    = False
    drawing      = False


cv2.namedWindow("GrabCut Interactive")
cv2.setMouseCallback("GrabCut Interactive", mouse_cb)

while True:
    display = draw_overlay(img)
    cv2.imshow("GrabCut Interactive", display)

    key = cv2.waitKey(20) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key == 13:  # ENTER
        apply_grabcut()
        result = get_result()
        cv2.imshow("GrabCut Result", result)
    elif key == ord('r'):
        reset()
        cv2.destroyWindow("GrabCut Result")
    elif key == ord('s'):
        result = get_result()
        path = os.path.join(OUTPUT_DIR, f"practice2_grabcut_{save_count}.png")
        cv2.imwrite(path, result)
        save_count += 1
        print(f"저장: {path}")

cv2.destroyAllWindows()
