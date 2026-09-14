"""Desktop interface for single-image and batch OCR keyword marking."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from ocr_core import SUPPORTED_IMAGE_TYPES, create_ocr, read_image, recognize_and_mark, write_image


class OCRLocatorApp:
    def __init__(self, root: TkinterDnD.Tk) -> None:
        self.root = root
        self.root.title("OCR 关键词识别与标注")
        self.root.geometry("1180x780")
        self.root.minsize(900, 650)
        self._ocr = None
        self._busy = False
        self._single_image: Path | None = None
        self._batch_folder: Path | None = None
        self._preview_refs: dict[str, ImageTk.PhotoImage] = {}

        self._configure_style()
        notebook = ttk.Notebook(root, padding=10)
        notebook.pack(fill="both", expand=True)
        self.single_tab = ttk.Frame(notebook, padding=14)
        self.batch_tab = ttk.Frame(notebook, padding=14)
        notebook.add(self.single_tab, text="单张图片")
        notebook.add(self.batch_tab, text="批量处理")
        self._build_single_tab()
        self._build_batch_tab()

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_single_tab(self) -> None:
        tab = self.single_tab
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)

        ttk.Label(tab, text="单图关键词标注", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        controls = ttk.Frame(tab)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="关键词：").grid(row=0, column=0, padx=(0, 6))
        self.single_query = ttk.Entry(controls)
        self.single_query.insert(0, "对象")
        self.single_query.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.single_button = ttk.Button(
            controls, text="开始识别并标注", style="Accent.TButton", command=self._start_single
        )
        self.single_button.grid(row=0, column=2)

        input_frame = ttk.LabelFrame(tab, text="原始图片（拖入此区域，或点击选择）", padding=8)
        output_frame = ttk.LabelFrame(tab, text="标注结果", padding=8)
        input_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        output_frame.grid(row=2, column=1, sticky="nsew", padx=(6, 0))
        input_frame.rowconfigure(0, weight=1)
        input_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.input_preview = tk.Label(
            input_frame, text="拖放图片到这里\n\n或单击选择图片", bg="#f4f6f8",
            fg="#52606d", font=("Microsoft YaHei UI", 13), cursor="hand2"
        )
        self.input_preview.grid(row=0, column=0, sticky="nsew")
        self.input_preview.bind("<Button-1>", lambda _event: self._choose_single_image())
        self.input_preview.drop_target_register(DND_FILES)
        self.input_preview.dnd_bind("<<Drop>>", self._on_drop)

        self.output_preview = tk.Label(
            output_frame, text="标注完成后将在这里显示", bg="#f4f6f8",
            fg="#7b8794", font=("Microsoft YaHei UI", 13)
        )
        self.output_preview.grid(row=0, column=0, sticky="nsew")
        self.single_status = ttk.Label(tab, text="请选择一张图片。", style="Hint.TLabel")
        self.single_status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_batch_tab(self) -> None:
        tab = self.batch_tab
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(5, weight=1)
        ttk.Label(tab, text="文件夹批量标注", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 16)
        )
        ttk.Label(tab, text="图片文件夹：").grid(row=1, column=0, sticky="w", pady=6)
        self.folder_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.folder_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=8
        )
        ttk.Button(tab, text="选择文件夹", command=self._choose_batch_folder).grid(row=1, column=2)

        ttk.Label(tab, text="关键词：").grid(row=2, column=0, sticky="w", pady=6)
        self.batch_query = ttk.Entry(tab)
        self.batch_query.insert(0, "对象")
        self.batch_query.grid(row=2, column=1, sticky="ew", padx=8)
        self.batch_button = ttk.Button(
            tab, text="开始批量处理", style="Accent.TButton", command=self._start_batch
        )
        self.batch_button.grid(row=2, column=2)

        ttk.Label(
            tab, text="支持 JPG、JPEG、PNG、BMP、WEBP；结果写入所选文件夹下的 outputs。",
            style="Hint.TLabel"
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 12))
        self.progress = ttk.Progressbar(tab, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        log_frame = ttk.LabelFrame(tab, text="处理记录", padding=8)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.batch_log = tk.Text(log_frame, state="disabled", wrap="word", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.batch_log.yview)
        self.batch_log.configure(yscrollcommand=scrollbar.set)
        self.batch_log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _choose_single_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if filename:
            self._set_single_image(Path(filename))

    def _on_drop(self, event: tk.Event) -> None:
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self._set_single_image(Path(paths[0]))

    def _set_single_image(self, path: Path) -> None:
        if path.suffix.lower() not in SUPPORTED_IMAGE_TYPES or not path.is_file():
            messagebox.showerror("无法打开", "请拖入受支持的图片文件。")
            return
        try:
            self._single_image = path
            self._show_preview(self.input_preview, path, "input")
            self.output_preview.configure(image="", text="标注完成后将在这里显示")
            self.single_status.configure(text=f"已选择：{path}")
        except Exception as error:
            messagebox.showerror("无法打开图片", str(error))

    def _show_preview(self, label: tk.Label, source: Path | object, key: str) -> None:
        if isinstance(source, Path):
            bgr = read_image(source)
        else:
            bgr = source
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(label.winfo_width() - 20, 380)
        height = max(label.winfo_height() - 20, 420)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._preview_refs[key] = photo
        label.configure(image=photo, text="")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.single_button.configure(state=state)
        self.batch_button.configure(state=state)

    def _get_ocr(self):
        if self._ocr is None:
            self._ocr = create_ocr()
        return self._ocr

    def _start_single(self) -> None:
        if self._busy:
            return
        if self._single_image is None:
            messagebox.showinfo("请选择图片", "请先拖入或选择一张图片。")
            return
        query = self.single_query.get().strip()
        if not query:
            messagebox.showinfo("请输入关键词", "关键词不能为空。")
            return
        self._set_busy(True)
        self.single_status.configure(text="正在加载模型并识别，请稍候……")
        threading.Thread(target=self._single_worker, args=(self._single_image, query), daemon=True).start()

    def _single_worker(self, path: Path, query: str) -> None:
        try:
            annotated, matches, line_count = recognize_and_mark(self._get_ocr(), path, query)
            output = Path(__file__).resolve().parent / "outputs" / f"{path.stem}_标记{path.suffix.lower()}"
            write_image(output, annotated)
            self.root.after(0, self._single_done, annotated, matches, line_count, output, query)
        except Exception as error:
            self.root.after(0, self._task_failed, "单图处理失败", error)

    def _single_done(self, annotated, matches, line_count: int, output: Path, query: str) -> None:
        self._show_preview(self.output_preview, annotated, "output")
        self._set_busy(False)
        if matches:
            self.single_status.configure(text=f"完成：找到 {len(matches)} 处“{query}”｜{output}")
        else:
            self.single_status.configure(text=f"完成：识别 {line_count} 行，未找到“{query}”｜{output}")

    def _choose_batch_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择待处理图片文件夹")
        if folder:
            self._batch_folder = Path(folder)
            self.folder_var.set(folder)

    def _start_batch(self) -> None:
        if self._busy:
            return
        if self._batch_folder is None or not self._batch_folder.is_dir():
            messagebox.showinfo("请选择文件夹", "请先选择包含图片的文件夹。")
            return
        query = self.batch_query.get().strip()
        if not query:
            messagebox.showinfo("请输入关键词", "关键词不能为空。")
            return
        images = sorted(
            path for path in self._batch_folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_TYPES
        )
        if not images:
            messagebox.showinfo("没有图片", "该文件夹中没有受支持的图片。")
            return
        self._set_busy(True)
        self.progress.configure(maximum=len(images), value=0)
        self._clear_log()
        self._append_log(f"共发现 {len(images)} 张图片，正在初始化 OCR……")
        threading.Thread(target=self._batch_worker, args=(images, query), daemon=True).start()

    def _batch_worker(self, images: list[Path], query: str) -> None:
        output_dir = images[0].parent / "outputs"
        found_images = 0
        failures = 0
        try:
            ocr = self._get_ocr()
            for index, path in enumerate(images, 1):
                try:
                    annotated, matches, _ = recognize_and_mark(ocr, path, query)
                    output = output_dir / f"{path.stem}_标记{path.suffix.lower()}"
                    write_image(output, annotated)
                    if matches:
                        found_images += 1
                    detail = f"找到 {len(matches)} 处" if matches else "未找到关键词"
                    self.root.after(0, self._batch_progress, index, f"[{index}/{len(images)}] {path.name}：{detail}")
                except Exception as error:
                    failures += 1
                    self.root.after(0, self._batch_progress, index, f"[{index}/{len(images)}] {path.name}：失败 - {error}")
            self.root.after(0, self._batch_done, len(images), found_images, failures, output_dir)
        except Exception as error:
            self.root.after(0, self._task_failed, "批量处理失败", error)

    def _batch_progress(self, index: int, text: str) -> None:
        self.progress.configure(value=index)
        self._append_log(text)

    def _batch_done(self, total: int, found: int, failures: int, output_dir: Path) -> None:
        self._set_busy(False)
        summary = f"处理完成：{total} 张，{found} 张找到关键词，{failures} 张失败。\n输出目录：{output_dir}"
        self._append_log(summary.replace("\n", "  "))
        messagebox.showinfo("批量处理完成", summary)

    def _task_failed(self, title: str, error: Exception) -> None:
        self._set_busy(False)
        messagebox.showerror(title, str(error))

    def _clear_log(self) -> None:
        self.batch_log.configure(state="normal")
        self.batch_log.delete("1.0", "end")
        self.batch_log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.batch_log.configure(state="normal")
        self.batch_log.insert("end", text + "\n")
        self.batch_log.see("end")
        self.batch_log.configure(state="disabled")


def main() -> None:
    root = TkinterDnD.Tk()
    OCRLocatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
