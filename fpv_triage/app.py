"""Desktop application for FPV clip triage."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .analyzer import AnalyzerConfig, FlightAnalysis
from .pipeline import run_triage


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class FpvTriageApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FPV Clip Triage")
        self.geometry("920x720")
        self.minsize(780, 600)

        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()

        self.source_var = ctk.StringVar()
        self.output_var = ctk.StringVar(
            value=str(Path.home() / "Videos" / "fpv-keepers")
        )
        self.min_flight_var = ctk.StringVar(value="10")
        self.threshold_var = ctk.StringVar(value="0.8")
        self.dry_run_var = ctk.BooleanVar(value=False)
        self.describe_var = ctk.BooleanVar(value=True)
        self.status_var = ctk.StringVar(value="Pick a source folder, then hit Start.")

        self._build()

    def _build(self) -> None:
        pad = {"padx": 16, "pady": 8}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        header = ctk.CTkLabel(
            self,
            text="FPV Clip Triage",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self,
            text="Scan an SD card, keep real flights, score speed, and AI-describe keepers for later editing.",
            text_color=("gray30", "gray70"),
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 8))

        form = ctk.CTkFrame(self)
        form.grid(row=2, column=0, sticky="ew", **pad)
        form.grid_columnconfigure(1, weight=1)

        self._path_row(form, 0, "Source (SD / folder)", self.source_var, self._browse_source)
        self._path_row(form, 1, "Output folder", self.output_var, self._browse_output)

        opts = ctk.CTkFrame(form, fg_color="transparent")
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        opts.grid_columnconfigure((0, 1, 2), weight=1)

        self._labeled_entry(opts, 0, "Min flight (sec)", self.min_flight_var)
        self._labeled_entry(opts, 1, "Motion threshold", self.threshold_var)

        checks = ctk.CTkFrame(opts, fg_color="transparent")
        checks.grid(row=0, column=2, rowspan=2, sticky="w", padx=12)
        ctk.CTkCheckBox(
            checks,
            text="AI descriptions (.txt notes)",
            variable=self.describe_var,
        ).pack(anchor="w")
        ctk.CTkCheckBox(
            checks,
            text="Dry run (analyze only, don't copy)",
            variable=self.dry_run_var,
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
        self.open_btn.pack(side="left")

        body = ctk.CTkFrame(self)
        body.grid(row=3, column=0, sticky="nsew", **pad)
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

    def _labeled_entry(
        self, parent: ctk.CTkFrame, column: int, label: str, variable: ctk.StringVar
    ) -> None:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=column, sticky="ew", padx=8)
        ctk.CTkLabel(box, text=label).pack(anchor="w")
        ctk.CTkEntry(box, textvariable=variable, width=120).pack(anchor="w", pady=(4, 0))

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

    def _set_running(self, running: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal")
        self.cancel_btn.configure(state="normal" if running else "disabled")

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
            min_flight = float(self.min_flight_var.get())
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Min flight and threshold must be numbers.")
            return

        if min_flight <= 0 or threshold < 0:
            messagebox.showerror("Settings", "Settings must be positive.")
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
        self._append_log(
            f"Keep rule: >= {min_flight:.0f}s flight  |  threshold={threshold}"
            + ("  |  AI descriptions ON" if self.describe_var.get() else "  |  AI descriptions OFF")
            + ("  |  DRY RUN" if self.dry_run_var.get() else "")
        )
        self._append_log("")

        config = AnalyzerConfig(
            min_flight_seconds=min_flight,
            motion_threshold=threshold,
        )
        dry_run = bool(self.dry_run_var.get())
        describe = bool(self.describe_var.get())

        def worker() -> None:
            try:
                if describe:
                    self.after(
                        0,
                        lambda: self.status_var.set(
                            "Loading AI model (first run downloads ~7GB)…"
                        ),
                    )
                summary = run_triage(
                    source,
                    output,
                    config,
                    dry_run=dry_run,
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
        if analysis.keep:
            dest = Path(copied_to).name if copied_to else "(dry run)"
            self._append_log(
                f"KEEP  {name}  ->  {dest}  "
                f"(speed {analysis.overall_speed_score:.1f}/10, "
                f"top {analysis.top_speed_score:.1f}/10)"
            )
            if analysis.description:
                preview = analysis.description.replace("\n", " ")
                if len(preview) > 160:
                    preview = preview[:157] + "..."
                self._append_log(f"      {preview}")
            if analysis.notes_path:
                self._append_log(f"      notes: {analysis.notes_path}")
        else:
            self._append_log(f"SKIP  {name}  ({analysis.reason})")

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
        if not cancelled and summary.kept and not self.dry_run_var.get():
            messagebox.showinfo(
                "Triage complete",
                f"Kept {summary.kept} of {summary.scanned} clips.\n\n"
                f"Copied to:\n{self.output_var.get()}\n\n"
                "Each keeper has a matching .txt notes file when AI descriptions are on.",
            )

    def _on_error(self, exc: Exception, tb: str) -> None:
        self._set_running(False)
        self.status_var.set("Failed.")
        self._append_log("")
        self._append_log(f"ERROR: {exc}")
        self._append_log(tb)
        messagebox.showerror("Triage failed", str(exc))


def main() -> None:
    app = FpvTriageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
