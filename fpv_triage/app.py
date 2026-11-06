"""Desktop application for FPV clip triage."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .analyzer import AnalyzerConfig, FlightAnalysis
from .catalog import NotesSearchIndex, rebuild_search_assets
from .describer import format_timing_line
from .pipeline import run_triage
from .scanner import delete_lrf_files, find_lrf_files


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class FpvTriageApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FPV Footage to Video")
        self.geometry("960x760")
        self.minsize(820, 640)

        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._search_index = NotesSearchIndex()

        self.source_var = ctk.StringVar()
        self.output_var = ctk.StringVar(
            value=str(Path.home() / "Videos" / "fpv-keepers")
        )
        self.threshold_var = ctk.StringVar(value="6.0")
        self.copy_videos_var = ctk.BooleanVar(value=True)
        self.describe_var = ctk.BooleanVar(value=True)
        self.status_var = ctk.StringVar(value="Pick a source folder, then hit Start.")
        self.search_query_var = ctk.StringVar()
        self.search_status_var = ctk.StringVar(
            value="Search keeper notes with AI (e.g. slow cruise near trees)."
        )

        self._build()

    def _build(self) -> None:
        pad = {"padx": 16, "pady": 8}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkLabel(
            self,
            text="FPV Footage to Video",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self,
            text="Triage SD-card flights, score speed, write movement notes, then AI-search keepers.",
            text_color=("gray30", "gray70"),
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 8))

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=2, column=0, sticky="nsew", **pad)
        triage = self.tabs.add("Triage")
        search = self.tabs.add("AI Search")
        self._build_triage_tab(triage)
        self._build_search_tab(search)

    def _build_triage_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        form = ctk.CTkFrame(tab)
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.grid_columnconfigure(1, weight=1)

        self._path_row(form, 0, "Source (SD / folder)", self.source_var, self._browse_source)
        self._path_row(form, 1, "Output folder", self.output_var, self._browse_output)

        opts = ctk.CTkFrame(form, fg_color="transparent")
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        opts.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            opts,
            text="Keep rule: 5s flight proven + 15s video still remaining",
            anchor="w",
            text_color=("gray30", "gray70"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

        thresh_box = ctk.CTkFrame(opts, fg_color="transparent")
        thresh_box.grid(row=1, column=0, sticky="w", padx=8)
        ctk.CTkLabel(thresh_box, text="Motion threshold").pack(anchor="w")
        ctk.CTkEntry(thresh_box, textvariable=self.threshold_var, width=120).pack(
            anchor="w", pady=(4, 0)
        )

        checks = ctk.CTkFrame(opts, fg_color="transparent")
        checks.grid(row=1, column=1, sticky="w", padx=12)
        ctk.CTkCheckBox(
            checks,
            text="AI descriptions (.txt + searchable index)",
            variable=self.describe_var,
        ).pack(anchor="w")
        ctk.CTkCheckBox(
            checks,
            text="Copy videos to output (off = notes only)",
            variable=self.copy_videos_var,
        ).pack(anchor="w", pady=(8, 0))

        buttons = ctk.CTkFrame(form, fg_color="transparent")
        buttons.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 4))

        self.start_btn = ctk.CTkButton(
            buttons, text="Start triage", width=140, command=self._start
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.cancel_btn = ctk.CTkButton(
            buttons,
            text="Cancel",
            width=100,
            fg_color=("gray65", "gray35"),
            state="disabled",
            command=self._request_cancel,
        )
        self.cancel_btn.pack(side="left", padx=(0, 8))

        self.open_btn = ctk.CTkButton(
            buttons,
            text="Open output",
            width=120,
            fg_color=("gray70", "gray30"),
            command=self._open_output,
        )
        self.open_btn.pack(side="left", padx=(0, 8))

        self.lrf_btn = ctk.CTkButton(
            buttons,
            text="Delete .LRF files",
            width=140,
            fg_color=("#a33", "#722"),
            hover_color=("#822", "#511"),
            command=self._delete_lrf_files,
        )
        self.lrf_btn.pack(side="left")

        body = ctk.CTkFrame(tab)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(2, weight=1)

        self.progress = ctk.CTkProgressBar(body)
        self.progress.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(body, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.log = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=13))
        self.log.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.log.insert("end", "Ready.\n")
        self.log.configure(state="disabled")

    def _build_search_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(tab)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        bar.grid_columnconfigure(0, weight=1)

        entry = ctk.CTkEntry(
            bar,
            textvariable=self.search_query_var,
            placeholder_text="slow cruise near trees, proximity pass, hover…",
        )
        entry.grid(row=0, column=0, sticky="ew", padx=(8, 8), pady=8)
        entry.bind("<Return>", lambda _e: self._run_search())

        ctk.CTkButton(bar, text="Search", width=100, command=self._run_search).grid(
            row=0, column=1, padx=(0, 8), pady=8
        )
        ctk.CTkButton(
            bar,
            text="Rebuild index",
            width=120,
            fg_color=("gray70", "gray30"),
            command=self._rebuild_index,
        ).grid(row=0, column=2, padx=(0, 8), pady=8)

        ctk.CTkLabel(tab, textvariable=self.search_status_var, anchor="w").grid(
            row=1, column=0, sticky="ew", padx=16, pady=(0, 4)
        )

        self.search_log = ctk.CTkTextbox(
            tab, font=ctk.CTkFont(family="Consolas", size=13)
        )
        self.search_log.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.search_log.insert(
            "end",
            "Uses the Output folder from the Triage tab.\n"
            "Searches keeper notes (keyword + meaning). Weak/unrelated hits are hidden.\n"
            "Example: slow cruise near trees\n"
            "If nothing matches, that word is not in the AI notes yet — re-triage with AI on.\n",
        )
        self.search_log.configure(state="disabled")

    def _path_row(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label: str,
        variable: ctk.StringVar,
        browse_cmd,
    ) -> None:
        ctk.CTkLabel(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(12, 8), pady=6
        )
        entry = ctk.CTkEntry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ctk.CTkButton(parent, text="Browse", width=90, command=browse_cmd).grid(
            row=row, column=2, padx=(8, 12), pady=6
        )

    def _browse_source(self) -> None:
        path = filedialog.askdirectory(title="Select SD card or footage folder")
        if path:
            self.source_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder on this PC")
        if path:
            self.output_var.set(path)

    def _open_output(self) -> None:
        output = Path(self.output_var.get().strip())
        if not output.exists():
            messagebox.showinfo("Output", "Output folder does not exist yet.")
            return
        try:
            import os

            os.startfile(output)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("Output", str(exc))

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _append_search(self, line: str) -> None:
        self.search_log.configure(state="normal")
        self.search_log.insert("end", line + "\n")
        self.search_log.see("end")
        self.search_log.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal")
        self.cancel_btn.configure(state="normal" if running else "disabled")
        self.lrf_btn.configure(state="disabled" if running else "normal")

    def _delete_lrf_files(self) -> None:
        source = Path(self.source_var.get().strip())
        if not source.exists() or not source.is_dir():
            messagebox.showerror(
                "Delete .LRF",
                "Choose a valid Source folder/drive first (SD card or footage folder).",
            )
            return

        try:
            files = find_lrf_files(source)
        except OSError as exc:
            messagebox.showerror("Delete .LRF", str(exc))
            return

        if not files:
            messagebox.showinfo("Delete .LRF", f"No .LRF files found under:\n{source}")
            return

        total_bytes = 0
        for path in files:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
        size_mb = total_bytes / (1024 * 1024)
        ok = messagebox.askyesno(
            "Delete .LRF",
            f"Delete {len(files)} DJI .LRF file(s) (~{size_mb:.1f} MB) under:\n"
            f"{source}\n\n"
            "These are low-res proxy files. Videos (.MP4) are not touched.",
        )
        if not ok:
            return

        self.lrf_btn.configure(state="disabled")
        self.status_var.set(f"Deleting {len(files)} .LRF file(s)…")

        def worker() -> None:
            try:
                deleted, failed = delete_lrf_files(source)
                self.after(0, lambda: self._on_lrf_deleted(deleted, failed))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_lrf_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_lrf_deleted(
        self, deleted: list[Path], failed: list[tuple[Path, str]]
    ) -> None:
        self.lrf_btn.configure(state="normal")
        self._append_log("")
        self._append_log(f"Deleted {len(deleted)} .LRF file(s).")
        for path in deleted[:20]:
            self._append_log(f"  - {path}")
        if len(deleted) > 20:
            self._append_log(f"  … and {len(deleted) - 20} more")
        for path, err in failed:
            self._append_log(f"  FAILED {path}: {err}")

        if failed:
            self.status_var.set(
                f"Deleted {len(deleted)} .LRF file(s); {len(failed)} failed."
            )
            messagebox.showwarning(
                "Delete .LRF",
                f"Deleted {len(deleted)} file(s).\n"
                f"{len(failed)} could not be removed (see log).",
            )
        else:
            self.status_var.set(f"Deleted {len(deleted)} .LRF file(s).")
            messagebox.showinfo(
                "Delete .LRF",
                f"Removed {len(deleted)} .LRF file(s) from the source folder.",
            )

    def _on_lrf_error(self, exc: Exception) -> None:
        self.lrf_btn.configure(state="normal")
        self.status_var.set("Delete .LRF failed.")
        messagebox.showerror("Delete .LRF", str(exc))

    def _request_cancel(self) -> None:
        self._cancel.set()
        self.status_var.set("Cancel requested… finishing current clip.")

    def _start(self) -> None:
        source = Path(self.source_var.get().strip())
        output = Path(self.output_var.get().strip())

        if not source.exists():
            messagebox.showerror("Source", "Choose a valid source folder or drive.")
            return
        if not self.output_var.get().strip():
            messagebox.showerror("Output", "Choose an output folder on this PC.")
            return

        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Motion threshold must be a number.")
            return

        if threshold < 0:
            messagebox.showerror("Settings", "Motion threshold must be >= 0.")
            return

        self._cancel.clear()
        self._set_running(True)
        self.progress.set(0)
        self.status_var.set("Starting…")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._append_log(f"Source: {source}")
        self._append_log(f"Output: {output}")
        copy_videos = bool(self.copy_videos_var.get())
        self._append_log(
            f"Keep rule: 5s flight + 15s remaining  |  threshold={threshold}"
            + ("  |  AI descriptions ON" if self.describe_var.get() else "  |  AI descriptions OFF")
            + ("  |  copy videos ON" if copy_videos else "  |  notes only (no video copy)")
        )
        self._append_log("")

        config = AnalyzerConfig(motion_threshold=threshold)
        describe = bool(self.describe_var.get())

        def worker() -> None:
            try:
                if describe:
                    self.after(
                        0,
                        lambda: self.status_var.set(
                            "Loading AI model (first run downloads weights)…"
                        ),
                    )
                summary = run_triage(
                    source,
                    output,
                    config,
                    copy_videos=copy_videos,
                    describe=describe,
                    should_cancel=self._cancel.is_set,
                    on_progress=lambda i, total, name: self.after(
                        0, lambda i=i, total=total, name=name: self._on_progress(i, total, name)
                    ),
                    on_result=lambda analysis, copied: self.after(
                        0,
                        lambda analysis=analysis, copied=copied: self._on_result(
                            analysis, copied
                        ),
                    ),
                )
                self.after(
                    0,
                    lambda: self._on_done(summary, cancelled=self._cancel.is_set()),
                )
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error(exc, tb))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_progress(self, index: int, total: int, name: str) -> None:
        fraction = 0.0 if total == 0 else index / total
        self.progress.set(fraction)
        self.status_var.set(f"{index}/{total}: {name}")

    def _on_result(self, analysis: FlightAnalysis, copied_to: str | None) -> None:
        name = Path(analysis.path).name
        timing = format_timing_line(analysis.timing)
        if analysis.keep:
            if copied_to:
                dest = Path(copied_to).name
            elif self.copy_videos_var.get():
                dest = "(copy pending/failed)"
            else:
                dest = "(notes only)"
            self._append_log(
                f"KEEP  {name}  ->  {dest}  "
                f"(speed {analysis.overall_speed_score:.1f}/10, "
                f"top {analysis.top_speed_score:.1f}/10)"
            )
            if analysis.environment_tag:
                self._append_log(f"      env: {analysis.environment_tag}")
            if analysis.people:
                self._append_log(f"      people: {analysis.people}")
            if analysis.action:
                self._append_log(f"      action: {analysis.action}")
            if analysis.description:
                preview = analysis.description.replace("\n", " ")
                if len(preview) > 160:
                    preview = preview[:157] + "..."
                self._append_log(f"      move: {preview}")
            if analysis.notes_path:
                self._append_log(f"      notes: {analysis.notes_path}")
            if timing:
                self._append_log(f"      time: {timing}")
        else:
            self._append_log(f"SKIP  {name}  ({analysis.reason})")
            if timing:
                self._append_log(f"      time: {timing}")

    def _on_done(self, summary, cancelled: bool) -> None:
        self._set_running(False)
        self.progress.set(1.0 if summary.scanned else 0.0)
        verb = "Cancelled" if cancelled else "Done"
        self.status_var.set(
            f"{verb}. Kept {summary.kept}/{summary.scanned}. Report: {summary.report_path}"
        )
        self._append_log("")
        self._append_log(
            f"{verb}. Kept {summary.kept}, skipped {summary.rejected}, "
            f"scanned {summary.scanned}."
        )
        self._append_log(f"Report: {summary.report_path}")
        if summary.kept and self.describe_var.get() and not cancelled:
            self._append_log(
                "Search index refreshed (clips_catalog.json + clips_search_index.npz)."
            )
            self.search_status_var.set(
                f"Indexed keepers in {self.output_var.get()}. Try the AI Search tab."
            )
        if not cancelled and summary.kept:
            if self.copy_videos_var.get():
                detail = f"Copied videos + notes to:\n{self.output_var.get()}"
            else:
                detail = (
                    f"Notes only (videos left on source) in:\n{self.output_var.get()}"
                )
            messagebox.showinfo(
                "Triage complete",
                f"Kept {summary.kept} of {summary.scanned} clips.\n\n"
                f"{detail}\n\n"
                "Notes are searchable on the AI Search tab.",
            )

    def _on_error(self, exc: Exception, tb: str) -> None:
        self._set_running(False)
        self.status_var.set("Failed.")
        self._append_log("")
        self._append_log(f"ERROR: {exc}")
        self._append_log(tb)
        messagebox.showerror("Triage failed", str(exc))

    def _rebuild_index(self) -> None:
        output = Path(self.output_var.get().strip())
        if not output.exists():
            messagebox.showerror("Search", "Output folder does not exist yet.")
            return

        self.search_status_var.set("Building search index…")

        def worker() -> None:
            try:
                catalog, count = rebuild_search_assets(output)
                self._search_index.load(output)
                self.after(
                    0,
                    lambda: self._on_index_done(catalog, count),
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_search_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_index_done(self, catalog: Path, count: int) -> None:
        self.search_status_var.set(f"Indexed {count} clip(s). Catalog: {catalog}")
        self.search_log.configure(state="normal")
        self.search_log.delete("1.0", "end")
        self.search_log.configure(state="disabled")
        self._append_search(f"Rebuilt index with {count} clip(s).")
        self._append_search(f"Catalog: {catalog}")

    def _run_search(self) -> None:
        output = Path(self.output_var.get().strip())
        query = self.search_query_var.get().strip()
        if not query:
            messagebox.showinfo("Search", "Type what kind of clip you want.")
            return
        if not output.exists():
            messagebox.showerror("Search", "Output folder does not exist yet.")
            return

        self.search_status_var.set(f"Searching: {query}")

        def worker() -> None:
            try:
                if not self._search_index.load(output):
                    rebuild_search_assets(output)
                    self._search_index.load(output)
                hits = self._search_index.search(query, top_k=10)
                self.after(0, lambda: self._on_search_results(query, hits))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_search_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_search_results(self, query: str, hits) -> None:
        self.search_log.configure(state="normal")
        self.search_log.delete("1.0", "end")
        self.search_log.configure(state="disabled")
        self._append_search(f"Query: {query}")
        if not hits:
            self._append_search(
                "No relevant matches — none of the notes mention that (or anything close).\n"
                "Re-run triage with AI descriptions on if you need object tags like chair/bench."
            )
            self.search_status_var.set("No relevant matches.")
            return

        self.search_status_var.set(f"{len(hits)} relevant match(es)")
        for rank, (record, score) in enumerate(hits, start=1):
            self._append_search("")
            self._append_search(
                f"{rank}. {record.file}  (score {score:.3f})  "
                f"speed {record.overall_speed_score:.1f}/{record.top_speed_score:.1f}"
            )
            self._append_search(f"   env: {record.environment}")
            if record.people:
                self._append_search(f"   people: {record.people}")
            if record.action:
                self._append_search(f"   action: {record.action}")
            move = record.movement.replace("\n", " ")
            if len(move) > 180:
                move = move[:177] + "..."
            self._append_search(f"   move: {move}")
            self._append_search(f"   video: {record.video_path}")

    def _on_search_error(self, exc: Exception) -> None:
        self.search_status_var.set("Search failed.")
        messagebox.showerror("Search failed", str(exc))


def main() -> None:
    app = FpvTriageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
