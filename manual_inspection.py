import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "real_screen_photos")
OUTPUT_DIR = os.path.join(INPUT_DIR, "rectified")
SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}


def my_sort(loc_x, loc_y):
    """
    对应Matlab的my_sort.m：对选取的4个角点排序（左上、右上、右下、左下）
    :param loc_x: 选点的x坐标列表
    :param loc_y: 选点的y坐标列表
    :return: 排序后的X(列向量)、Y(列向量)
    """
    # 修正中值计算（核心：max+min后除以2）
    x_mid = (np.max(loc_x) + np.min(loc_x)) / 2
    y_mid = (np.max(loc_y) + np.min(loc_y)) / 2

    # 初始化排序后的坐标（左上、右上、右下、左下）
    X = np.zeros((4, 1))  # 列向量，对应Matlab的X'
    Y = np.zeros((4, 1))

    for i in range(4):
        x = loc_x[i]
        y = loc_y[i]
        if x < x_mid and y < y_mid:
            # 左上
            X[0, 0] = x
            Y[0, 0] = y
        elif x > x_mid and y < y_mid:
            # 右上
            X[1, 0] = x
            Y[1, 0] = y
        elif x > x_mid and y > y_mid:
            # 右下
            X[2, 0] = x
            Y[2, 0] = y
        elif x < x_mid and y > y_mid:
            # 左下
            X[3, 0] = x
            Y[3, 0] = y

    return X, Y


def my_pres_trans(img, X, Y):
    """
    对应Matlab的my_pres_trans.m：透视变换核心逻辑
    :param img: 原始图像（BGR格式）
    :param X: 排序后的x坐标列向量
    :param Y: 排序后的y坐标列向量
    :return: 按所选区域原生像素尺寸完成透视矫正后的图像
    """
    # 1. 源图像4个角点（已排序：左上、右上、右下、左下）
    src_pts = np.hstack((X, Y)).astype(np.float32)  # 拼接为N×2的浮点数组

    # 2. 根据四条边在原始照片中的实际像素长度确定输出尺寸。
    width_top = np.linalg.norm(src_pts[1] - src_pts[0])
    width_bottom = np.linalg.norm(src_pts[2] - src_pts[3])
    height_left = np.linalg.norm(src_pts[3] - src_pts[0])
    height_right = np.linalg.norm(src_pts[2] - src_pts[1])
    side_length = max(
        2,
        int(round(max(width_top, width_bottom, height_left, height_right))),
    )
    out_w = side_length
    out_h = side_length

    target_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    # 3. 计算透视变换矩阵（对应Matlab的fitgeotrans）
    M = cv2.getPerspectiveTransform(src_pts, target_pts)

    # 4. 执行透视变换（对应Matlab的imtransform/imwarp）
    # 注意：OpenCV的warpPerspective参数是(dst_width, dst_height)
    img_warp = cv2.warpPerspective(
        img,
        M,
        (out_w, out_h),
        borderMode=cv2.BORDER_CONSTANT,  # 边界填充方式
        borderValue=(255, 255, 255)  # 白色填充
    )

    return img_warp


def select_corners_on_original(img, image_name, current_index, total_images):
    """在可缩放、可滚动的原图画布上选择四个角点。"""
    selected_points = []
    action = {"value": "stop"}
    zoom = {"value": 1.0}
    image_offset = {"x": 0.0, "y": 0.0}
    min_zoom = 0.02
    max_zoom = 1.0

    root = tk.Tk()
    root.title(
        f"[{current_index}/{total_images}] {image_name} - "
        "Select 4 Corners (TL -> TR -> BR -> BL)"
    )
    root.geometry("1400x900")

    def bring_to_front():
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.after(500, lambda: root.attributes("-topmost", False))

    progress_prefix = f"[{current_index}/{total_images}] {image_name}"
    status = tk.StringVar(value=f"{progress_prefix}；当前 0/4")
    tk.Label(root, textvariable=status, anchor="w").pack(fill=tk.X, padx=8, pady=4)

    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(canvas_frame, background="black", highlightthickness=0)
    x_scrollbar = tk.Scrollbar(
        canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview
    )
    y_scrollbar = tk.Scrollbar(
        canvas_frame, orient=tk.VERTICAL, command=canvas.yview
    )
    canvas.configure(
        xscrollcommand=x_scrollbar.set,
        yscrollcommand=y_scrollbar.set,
    )
    canvas.grid(row=0, column=0, sticky="nsew")
    y_scrollbar.grid(row=0, column=1, sticky="ns")
    x_scrollbar.grid(row=1, column=0, sticky="ew")
    canvas_frame.rowconfigure(0, weight=1)
    canvas_frame.columnconfigure(0, weight=1)

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    source_image = Image.fromarray(rgb_image)
    image_item = canvas.create_image(
        0, 0, anchor=tk.NW, tags="source_image"
    )
    zoom_text = tk.StringVar(value="缩放：100%")

    def redraw_points():
        canvas.delete("selected_point")
        for index, (x, y) in enumerate(selected_points, start=1):
            display_x = image_offset["x"] + x * zoom["value"]
            display_y = image_offset["y"] + y * zoom["value"]
            radius = 7
            canvas.create_oval(
                display_x - radius,
                display_y - radius,
                display_x + radius,
                display_y + radius,
                outline="yellow",
                fill="red",
                width=2,
                tags="selected_point",
            )
            canvas.create_text(
                display_x + 12,
                display_y - 12,
                text=str(index),
                fill="yellow",
                font=("Arial", 14, "bold"),
                anchor=tk.SW,
                tags="selected_point",
            )
        status.set(
            f"{progress_prefix}；请按左上、右上、右下、左下的顺序点击；"
            f"当前 {len(selected_points)}/4"
        )

    def render_image(new_zoom, preserve_center=True):
        old_zoom = zoom["value"]
        view_width = max(canvas.winfo_width(), 1)
        view_height = max(canvas.winfo_height(), 1)
        if preserve_center:
            center_x = (
                canvas.canvasx(view_width / 2) - image_offset["x"]
            ) / old_zoom
            center_y = (
                canvas.canvasy(view_height / 2) - image_offset["y"]
            ) / old_zoom
        else:
            center_x = img.shape[1] / 2
            center_y = img.shape[0] / 2

        new_zoom = max(min_zoom, min(float(new_zoom), max_zoom))
        scaled_width = max(1, int(round(img.shape[1] * new_zoom)))
        scaled_height = max(1, int(round(img.shape[0] * new_zoom)))
        resampling = (
            Image.Resampling.LANCZOS
            if new_zoom < 1.0
            else Image.Resampling.BICUBIC
        )
        display_image = source_image.resize(
            (scaled_width, scaled_height),
            resampling,
        )
        tk_image = ImageTk.PhotoImage(display_image)
        canvas.itemconfigure(image_item, image=tk_image)
        image_offset["x"] = max((view_width - scaled_width) / 2, 0.0)
        image_offset["y"] = max((view_height - scaled_height) / 2, 0.0)
        canvas.coords(image_item, image_offset["x"], image_offset["y"])
        canvas.image = tk_image
        scroll_width = max(scaled_width, view_width)
        scroll_height = max(scaled_height, view_height)
        canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
        zoom["value"] = new_zoom
        zoom_text.set(f"缩放：{new_zoom * 100:.0f}%")
        redraw_points()

        root.update_idletasks()
        view_width = max(canvas.winfo_width(), 1)
        view_height = max(canvas.winfo_height(), 1)
        if scaled_width > view_width:
            left = center_x * new_zoom - view_width / 2
            canvas.xview_moveto(max(0.0, min(1.0, left / scroll_width)))
        else:
            canvas.xview_moveto(0.0)
        if scaled_height > view_height:
            top = center_y * new_zoom - view_height / 2
            canvas.yview_moveto(max(0.0, min(1.0, top / scroll_height)))
        else:
            canvas.yview_moveto(0.0)

    def fit_image():
        root.update_idletasks()
        fit_zoom = min(
            canvas.winfo_width() / img.shape[1],
            canvas.winfo_height() / img.shape[0],
            1.0,
        )
        render_image(fit_zoom, preserve_center=False)

    def actual_size():
        render_image(1.0)

    def zoom_in():
        render_image(zoom["value"] * 1.25)

    def zoom_out():
        render_image(zoom["value"] / 1.25)

    def on_left_click(event):
        if len(selected_points) >= 4:
            return
        x = int(round(
            (canvas.canvasx(event.x) - image_offset["x"]) / zoom["value"]
        ))
        y = int(round(
            (canvas.canvasy(event.y) - image_offset["y"]) / zoom["value"]
        ))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            selected_points.append((x, y))
            redraw_points()

    def undo_last():
        if selected_points:
            selected_points.pop()
            redraw_points()

    def reset_points():
        selected_points.clear()
        redraw_points()

    def confirm_points():
        if len(selected_points) != 4:
            status.set(f"需要选择 4 个角点，当前为 {len(selected_points)} 个")
            return
        action["value"] = "confirm"
        root.destroy()

    def skip_current():
        action["value"] = "skip"
        root.destroy()

    def stop_batch():
        action["value"] = "stop"
        root.destroy()

    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-event.delta / 120), "units")

    def on_shift_mouse_wheel(event):
        canvas.xview_scroll(int(-event.delta / 120), "units")

    def on_zoom_wheel(event):
        if event.delta > 0:
            zoom_in()
        else:
            zoom_out()
        return "break"

    def start_pan(event):
        canvas.scan_mark(event.x, event.y)

    def drag_pan(event):
        canvas.scan_dragto(event.x, event.y, gain=1)

    canvas.bind("<Button-1>", on_left_click)
    canvas.bind("<Button-2>", start_pan)
    canvas.bind("<B2-Motion>", drag_pan)
    canvas.bind("<MouseWheel>", on_mouse_wheel)
    canvas.bind("<Shift-MouseWheel>", on_shift_mouse_wheel)
    canvas.bind("<Control-MouseWheel>", on_zoom_wheel)
    root.bind("<BackSpace>", lambda event: undo_last())
    root.bind("<Escape>", lambda event: skip_current())
    root.bind("<Return>", lambda event: confirm_points())
    root.bind("<KP_Enter>", lambda event: confirm_points())
    root.bind("<Key-plus>", lambda event: zoom_in())
    root.bind("<Key-minus>", lambda event: zoom_out())

    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, padx=8, pady=8)
    tk.Button(button_frame, text="适合窗口", command=fit_image).pack(side=tk.LEFT)
    tk.Button(button_frame, text="100%", command=actual_size).pack(
        side=tk.LEFT, padx=(8, 0)
    )
    tk.Button(button_frame, text="放大 +", command=zoom_in).pack(
        side=tk.LEFT, padx=(8, 0)
    )
    tk.Button(button_frame, text="缩小 -", command=zoom_out).pack(
        side=tk.LEFT, padx=(8, 0)
    )
    tk.Label(button_frame, textvariable=zoom_text).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="撤销", command=undo_last).pack(side=tk.LEFT)
    tk.Button(button_frame, text="重选", command=reset_points).pack(
        side=tk.LEFT, padx=8
    )
    tk.Button(button_frame, text="结束批处理", command=stop_batch).pack(
        side=tk.RIGHT
    )
    tk.Button(button_frame, text="跳过当前", command=skip_current).pack(
        side=tk.RIGHT, padx=(0, 8)
    )
    tk.Button(button_frame, text="确认四点", command=confirm_points).pack(
        side=tk.RIGHT, padx=8
    )

    root.protocol("WM_DELETE_WINDOW", stop_batch)
    root.after(0, bring_to_front)
    root.after(100, fit_image)
    root.mainloop()
    return action["value"], selected_points


def is_path_inside(path, parent):
    """判断 path 是否位于 parent 目录内（包含 parent 本身）。"""
    try:
        path = os.path.normcase(os.path.abspath(path))
        parent = os.path.normcase(os.path.abspath(parent))
        return os.path.commonpath([path, parent]) == parent
    except ValueError:
        return False


def choose_input_directory():
    """让用户选择本次需要进行透视矫正的距离目录。"""
    os.makedirs(INPUT_DIR, exist_ok=True)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    selected_dir = None
    while True:
        folder = filedialog.askdirectory(
            parent=root,
            title="选择需要透视矫正的距离文件夹",
            initialdir=INPUT_DIR,
            mustexist=True,
        )
        if not folder:
            break

        folder = os.path.abspath(folder)
        if not is_path_inside(folder, INPUT_DIR):
            messagebox.showerror(
                "目录无效",
                "请选择 real_screen_photos 目录下的距离文件夹。",
                parent=root,
            )
            continue
        if is_path_inside(folder, OUTPUT_DIR):
            messagebox.showerror(
                "目录无效",
                "不能选择 rectified 输出目录，请选择 20cm、30cm 等原始照片目录。",
                parent=root,
            )
            continue

        selected_dir = folder
        break

    root.destroy()
    return selected_dir


def collect_input_images(selected_dir):
    """递归收集所选距离目录中的图片，并排除已有的矫正结果。"""
    image_paths = []
    for current_dir, dir_names, file_names in os.walk(selected_dir):
        dir_names[:] = [
            name for name in dir_names if name.lower() != "rectified"
        ]
        for file_name in file_names:
            stem, extension = os.path.splitext(file_name)
            if (
                extension.lower() in SUPPORTED_EXTENSIONS
                and not stem.lower().endswith("_re")
            ):
                image_paths.append(os.path.join(current_dir, file_name))
    return sorted(
        image_paths,
        key=lambda path: os.path.relpath(path, selected_dir).lower(),
    )


def main():
    """
    批量执行手工透视矫正，并保存到 real_screen_photos/rectified。
    """
    selected_dir = choose_input_directory()
    if selected_dir is None:
        print("未选择目录，已取消透视矫正。")
        return

    image_paths = collect_input_images(selected_dir)
    if not image_paths:
        print(f"未在所选目录中找到支持的图片：{selected_dir}")
        return

    selected_relative_dir = os.path.relpath(selected_dir, INPUT_DIR)
    selected_output_dir = (
        OUTPUT_DIR
        if selected_relative_dir == "."
        else os.path.join(OUTPUT_DIR, selected_relative_dir)
    )
    os.makedirs(selected_output_dir, exist_ok=True)
    processed = 0
    skipped = 0
    failed = 0
    stopped = False

    print(f"已选择目录：{selected_dir}")
    print(f"发现 {len(image_paths)} 张图片")
    print(f"矫正结果将保存到：{selected_output_dir}")

    for current_index, file_path in enumerate(image_paths, start=1):
        file_name = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, INPUT_DIR)
        print(f"[{current_index}/{len(image_paths)}] 正在处理：{relative_path}")

        # OpenCV 默认以 BGR 格式读取图片。
        img_ori = cv2.imread(file_path)
        if img_ori is None:
            failed += 1
            print(f"  读取失败，已跳过：{file_path}")
            continue

        action, selected_points = select_corners_on_original(
            img_ori,
            image_name=relative_path,
            current_index=current_index,
            total_images=len(image_paths),
        )
        if action == "stop":
            stopped = True
            print("用户结束了批处理")
            break
        if action == "skip":
            skipped += 1
            print(f"  已跳过：{file_name}")
            continue
        if len(selected_points) != 4:
            failed += 1
            print(f"  角点数量异常，已跳过：{file_name}")
            continue

        loc_x = np.array(
            [max(1, int(np.floor(point[0]))) for point in selected_points]
        )
        loc_y = np.array(
            [max(1, int(np.floor(point[1]))) for point in selected_points]
        )
        X, Y = my_sort(loc_x, loc_y)
        rectified = my_pres_trans(img_ori, X, Y)

        stem = os.path.splitext(file_name)[0]
        relative_dir = os.path.dirname(relative_path)
        save_dir = os.path.join(OUTPUT_DIR, relative_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{stem}_re.png")
        if not cv2.imwrite(save_path, rectified):
            failed += 1
            print(f"  保存失败：{save_path}")
            continue

        processed += 1
        print(
            f"  已保存：{save_path} "
            f"({rectified.shape[1]}×{rectified.shape[0]})"
        )

    status = "已提前结束" if stopped else "已完成"
    print(
        f"批处理{status}：成功 {processed}，跳过 {skipped}，失败 {failed}，"
        f"总计 {len(image_paths)}"
    )


if __name__ == "__main__":
    main()
