"""Tkinter desktop interface for live FSL landmark-fingerprint matching."""
from __future__ import annotations

import base64
import queue
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2

from .live_recognition import LiveRecognizer


class WebcamApplication:
    def __init__(self, root: tk.Tk, project_root: str | Path | None = None):
        self.root = root
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.root.title("Filipino Sign Language Recognition")
        self.root.minsize(760, 650)
        self.events: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event: threading.Event | None = None
        self.camera_thread: threading.Thread | None = None
        self.camera_source_var = tk.StringVar(value="Computer webcam")
        self.phone_url_var = tk.StringVar(value="")
        self.recognizer = None
        self.photo = None
        self._build_ui()
        try:
            self.recognizer = LiveRecognizer(self.project_root, stability_frames=3)
            self.status_var.set("Model ready. Select Start Camera.")
            self.start_button.configure(state="normal")
        except Exception as exc:
            self.status_var.set(f"Model unavailable: {exc}")
            self.start_button.configure(state="disabled")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self._poll_events)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Filipino Sign Language Recognition",
                  font=("Segoe UI", 18, "bold")).pack(anchor="w")
        self.status_var = tk.StringVar(value="Loading recognition model…")
        ttk.Label(outer, textvariable=self.status_var, wraplength=720).pack(anchor="w", pady=(6, 10))
        self.image_label = ttk.Label(outer, text="Camera is stopped", anchor="center",
                                     background="#20252b", foreground="white")
        self.image_label.pack(fill="both", expand=True, pady=(0, 10))
        source_row = ttk.Frame(outer)
        source_row.pack(fill="x", pady=(0, 8))
        ttk.Label(source_row, text="Camera source:").pack(side="left")
        self.source_selector = ttk.Combobox(source_row, textvariable=self.camera_source_var,
                                            values=("Computer webcam", "Phone camera stream"),
                                            state="readonly", width=22)
        self.source_selector.pack(side="left", padx=(6, 12))
        ttk.Label(source_row, text="Phone stream URL:").pack(side="left")
        self.phone_url_entry = ttk.Entry(source_row, textvariable=self.phone_url_var, state="disabled")
        self.phone_url_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(outer, text="For phone input, use a camera-streaming app on the phone and select its front or back camera there. Keep both devices on the same Wi-Fi.",
                  wraplength=720).pack(anchor="w", pady=(0, 8))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="Start Camera", command=self.start_camera, state="disabled")
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 16))
        self.source_selector.bind("<<ComboboxSelected>>", self._update_source_controls)
        self.latency_var = tk.StringVar(value="Processing latency: —")
        ttk.Label(controls, textvariable=self.latency_var).pack(side="left")
        self.detection_var = tk.StringVar(value="Detection status: Camera stopped")
        ttk.Label(outer, textvariable=self.detection_var).pack(anchor="w", pady=(8, 0))
        results = ttk.LabelFrame(outer, text="Recognition", padding=10)
        results.pack(fill="x", pady=(12, 0))
        self.hands_var = tk.StringVar(value="Hands detected: 0")
        ttk.Label(results, textvariable=self.hands_var, wraplength=700).pack(anchor="w")
        self.gesture_var = tk.StringVar(value="Gesture: —")
        self.english_var = tk.StringVar(value="English: —")
        self.cebuano_var = tk.StringVar(value="Cebuano: —")
        ttk.Label(results, textvariable=self.gesture_var, font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(results, textvariable=self.english_var).pack(anchor="w", pady=(3, 0))
        ttk.Label(results, textvariable=self.cebuano_var, wraplength=700).pack(anchor="w", pady=(3, 0))

    def start_camera(self):
        if self.recognizer is None or (self.camera_thread and self.camera_thread.is_alive()):
            return
        self.stop_event = threading.Event()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Opening webcam…")
        source = self.camera_source_var.get()
        stream_url = self.phone_url_var.get().strip()
        if source == "Phone camera stream" and not stream_url:
            self.status_var.set("Enter the phone camera stream URL first.")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return
        self.camera_thread = threading.Thread(target=self._camera_loop, args=(source, stream_url),
                                              daemon=True, name="fsl-webcam")
        self.camera_thread.start()

    def _update_source_controls(self, _event=None):
        state = "normal" if self.camera_source_var.get() == "Phone camera stream" else "disabled"
        self.phone_url_entry.configure(state=state)

    def stop_camera(self):
        if self.stop_event is not None:
            self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping webcam…")

    def _camera_loop(self, source="Computer webcam", stream_url=""):
        if source == "Phone camera stream":
            try:
                cap = open_phone_stream(stream_url)
            except ValueError as exc:
                self._post({"type": "error", "message": str(exc)})
                self._post({"type": "stopped"})
                return
            selected_index = None
        else:
            cap, selected_index = open_webcam()
        if cap is None:
            message = ("Could not open the phone stream. Check the URL, phone streaming app, network, and that both devices use the same Wi-Fi."
                       if source == "Phone camera stream" else
                       "Could not open a computer webcam (tried indexes 0, 1, and 2). Check camera permissions, connection, and whether another app is using it.")
            self._post({"type": "error", "message": message})
            self._post({"type": "stopped"})
            return
        message = f"Phone stream is running: {stream_url}" if source == "Phone camera stream" else f"Computer camera {selected_index} is running."
        self._post({"type": "status", "message": message})
        try:
            while self.stop_event is not None and not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._post({"type": "error", "message": "The camera source opened, but no frames could be read."})
                    break
                if source == "Computer webcam":
                    frame = cv2.flip(frame, 1)
                result = self.recognizer.process_frame(frame)
                self._post({"type": "frame", "result": result})
        except Exception as exc:
            self._post({"type": "error", "message": f"Camera or recognition error: {exc}"})
        finally:
            cap.release()
            self._post({"type": "stopped"})

    def _post(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event["type"] == "frame":
                    self._show_result(event["result"])
                elif event["type"] == "status":
                    self.status_var.set(event["message"])
                elif event["type"] == "error":
                    self.status_var.set(event["message"])
                    self.detection_var.set("Detection status: Camera or recognition error")
                elif event["type"] == "stopped":
                    self.start_button.configure(state="normal" if self.recognizer else "disabled")
                    self.stop_button.configure(state="disabled")
                    if self.status_var.get() == "Stopping webcam…":
                        self.status_var.set("Camera stopped.")
                    self.detection_var.set("Detection status: Camera stopped")
        except queue.Empty:
            pass
        self.root.after(40, self._poll_events)

    def _show_result(self, result):
        frame = result["frame"]
        height, width = frame.shape[:2]
        max_width, max_height = 800, 450
        scale = min(max_width / width, max_height / height, 1.0)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        ok, encoded = cv2.imencode(".png", frame)
        if ok:
            self.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))
            self.image_label.configure(image=self.photo, text="")
        self.latency_var.set(f"Processing latency: {result['latency_ms']:.1f} ms")
        self.detection_var.set(f"Detection status: {result['detection_status']}")
        sides = ", ".join(hand["side"] for hand in result.get("hands", []))
        self.hands_var.set(f"Hands detected: {result.get('detected_hand_count', 0)}" + (f" ({sides})" if sides else ""))
        status = result["status"]
        if status == "no_hand":
            self.gesture_var.set("Gesture: No hand detected")
            self.english_var.set("English: —")
            self.cebuano_var.set("Cebuano: —")
        elif status == "invalid_hand":
            self.gesture_var.set("Gesture: Hand landmarks unusable")
            self.english_var.set("English: —")
            self.cebuano_var.set(f"Cebuano: {result.get('error', 'Invalid landmarks')}")
        elif status == "stabilizing":
            self.gesture_var.set(f"Gesture: Stabilizing ({result['raw_label']})…")
            self.english_var.set("English: —")
            self.cebuano_var.set("Cebuano: —")
        elif status == "unknown":
            self.gesture_var.set("Gesture: UNKNOWN")
            self.english_var.set("English: —")
            self.cebuano_var.set("Cebuano: No supported class matched within the saved threshold.")
        elif status == "two_hand_unsupported":
            self.gesture_var.set("Gesture: UNKNOWN (two-hand gesture model not available)")
            self.english_var.set("English: —")
            self.cebuano_var.set("Cebuano: —")
            details = []
            for hand in result["hands"]:
                if hand["distance"] is None:
                    details.append(f"{hand['side']}: landmarks unusable")
                else:
                    details.append(f"{hand['side']} individual-hand candidate: {hand['candidate_label']} ({hand['distance']:.3f})")
            self.detection_var.set(result["detection_status"] + ". Per-hand matches are diagnostic only.")
            self.hands_var.set("Hands detected: 2; " + "; ".join(details) + ". These do not classify the combined gesture.")
        else:
            label = result["label"]
            mapping = self.recognizer.translation_for(label)
            if mapping is None:
                self.gesture_var.set("Gesture: UNKNOWN")
                self.english_var.set("English: —")
                self.cebuano_var.set("Cebuano: —")
                return
            self.gesture_var.set(f"Gesture: {label}  (distance {result['distance']:.3f})")
            self.english_var.set(f"English: {mapping['english'] or 'No mapping defined'}")
            cebuano = mapping["cebuano"] or "No mapping defined"
            note = " — translation pending review" if mapping["review_status"] == "REVIEW REQUIRED" else ""
            self.cebuano_var.set(f"Cebuano: {cebuano}{note}")

    def close(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        if self.recognizer is not None:
            self.recognizer.close()
            self.recognizer = None
        self.root.destroy()


def main():
    root = tk.Tk()
    WebcamApplication(root)
    root.mainloop()


def open_webcam(indexes=(0, 1, 2)):
    """Try common webcam indexes and release every unsuccessful capture."""
    backends = [cv2.CAP_ANY]
    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for index in indexes:
        for backend in backends:
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                return capture, index
            capture.release()
    return None, None


def open_phone_stream(url: str):
    """Open an HTTP/HTTPS/RTSP stream exposed by a phone camera app."""
    value = url.strip()
    if not value:
        raise ValueError("Enter the phone camera stream URL first.")
    if not value.lower().startswith(("http://", "https://", "rtsp://", "rtsps://")):
        raise ValueError("Phone stream URL must start with http://, https://, rtsp://, or rtsps://.")
    capture = cv2.VideoCapture(value, cv2.CAP_ANY)
    if capture.isOpened():
        return capture
    capture.release()
    return None


if __name__ == "__main__":
    main()
