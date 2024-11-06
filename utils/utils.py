# utils/utils.py

import time
import tkinter as tk  # Corrected import

def update_progress(app_instance, progress, total_steps, start_time):
    """
    Update the progress bar and estimated time remaining.

    Parameters:
    - app_instance: Instance of the MLModelBuilderApp
    - progress: Current progress value
    - total_steps: Total number of steps
    - start_time: Start time of the task
    """
    percentage = int((progress / total_steps) * 100)
    elapsed_time = time.time() - start_time
    if progress > 0:
        estimated_total_time = (elapsed_time / progress) * total_steps
        remaining_time = estimated_total_time - elapsed_time
        remaining_time_formatted = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    else:
        remaining_time_formatted = "Calculating..."

    app_instance.progress_bar.set(progress / total_steps)
    app_instance.progress_label.configure(text=f"Progress: {percentage}% - Estimated time remaining: {remaining_time_formatted}")
    app_instance.root.update_idletasks()

def create_tooltip(widget, text):
    """
    Create a tooltip for a given widget.

    Parameters:
    - widget: The widget to attach the tooltip to
    - text: The text to display in the tooltip
    """
    tooltip = None

    def enter(event):
        nonlocal tooltip
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + 20
        tooltip = tk.Toplevel(widget)
        tooltip.overrideredirect(True)
        tooltip.geometry(f"+{x}+{y}")
        label = tk.Label(tooltip, text=text, background="black", relief="solid", borderwidth=1)
        label.pack()

    def leave(event):
        nonlocal tooltip
        if tooltip:
            tooltip.destroy()
            tooltip = None

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)
