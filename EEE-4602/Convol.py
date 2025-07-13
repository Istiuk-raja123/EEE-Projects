import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import time
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as patches

class ConvolutionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌊 Buffer Overflow's Convolution Studio")
        self.root.geometry("1400x900")
        
        self.theme = tk.StringVar(value="dark")
        self.setup_themes()
        
        self.mode = tk.StringVar(value="discrete")
        self.input_signal = None
        self.impulse_response = None
        self.input_indices = None
        self.impulse_indices = None
        self.output_signal = None
        self.output_indices = None
        self.animation_running = False
        self.animation_speed = 150
        self.animation_quality = tk.StringVar(value="high")
        
        self.current_frame = 0
        self.total_frames = 0
        self.glow_intensity = 0.0
        self.particle_effects = []
        
        self.setup_styles()
        self.setup_gui()
        self.apply_theme()
        
    def setup_themes(self):
        self.themes = {
            "dark": {
                "bg": "#1a1a1a", "fg": "#e0e0e0", "card_bg": "#2d2d2d", "accent": "#00ffff",
                "secondary": "#ff6b6b", "success": "#51cf66", "warning": "#ffd43b",
                "entry_bg": "#3a3a3a", "entry_fg": "#ffffff", "button_bg": "#4a4a4a",
                "button_fg": "#ffffff", "plot_bg": "#1e1e1e", "plot_fg": "#ffffff",
                "grid_color": "#404040"
            },
            "light": {
                "bg": "#f8f9fa", "fg": "#2d3436", "card_bg": "#ffffff", "accent": "#0984e3",
                "secondary": "#e17055", "success": "#00b894", "warning": "#fdcb6e",
                "entry_bg": "#ffffff", "entry_fg": "#2d3436", "button_bg": "#ddd",
                "button_fg": "#2d3436", "plot_bg": "#ffffff", "plot_fg": "#2d3436",
                "grid_color": "#e0e0e0"
            }
        }
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
    def apply_theme(self):
        current_theme = self.themes[self.theme.get()]
        self.root.configure(bg=current_theme["bg"])
        
        self.style.configure('Card.TFrame', background=current_theme["card_bg"], relief='flat', borderwidth=1)
        self.style.configure('Title.TLabel', background=current_theme["card_bg"], foreground=current_theme["accent"], font=('Segoe UI', 12, 'bold'))
        self.style.configure('Heading.TLabel', background=current_theme["card_bg"], foreground=current_theme["fg"], font=('Segoe UI', 10, 'bold'))
        self.style.configure('Body.TLabel', background=current_theme["card_bg"], foreground=current_theme["fg"], font=('Segoe UI', 9))
        self.style.configure('Accent.TButton', background=current_theme["accent"], foreground=current_theme["plot_bg"], font=('Segoe UI', 10, 'bold'), focuscolor='none')
        self.style.configure('Secondary.TButton', background=current_theme["secondary"], foreground=current_theme["plot_bg"], font=('Segoe UI', 10, 'bold'), focuscolor='none')
        self.style.configure('Success.TButton', background=current_theme["success"], foreground=current_theme["plot_bg"], font=('Segoe UI', 10, 'bold'), focuscolor='none')
        
        plt.style.use('dark_background' if self.theme.get() == 'dark' else 'default')
        
        if hasattr(self, 'fig'):
            self.fig.patch.set_facecolor(current_theme["plot_bg"])
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_facecolor(current_theme["plot_bg"])
                ax.tick_params(colors=current_theme["plot_fg"])
                ax.xaxis.label.set_color(current_theme["plot_fg"])
                ax.yaxis.label.set_color(current_theme["plot_fg"])
                ax.title.set_color(current_theme["accent"])
                ax.grid(True, color=current_theme["grid_color"], alpha=0.3)
            self.canvas.draw()
        
    def setup_gui(self):
        main_container = tk.Frame(self.root, bg=self.themes[self.theme.get()]["bg"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        header_frame = tk.Frame(main_container, bg=self.themes[self.theme.get()]["bg"])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(header_frame, text="🌊 Buffer Overflow's Convolution Studio", 
                              font=('Segoe UI', 20, 'bold'), bg=self.themes[self.theme.get()]["bg"],
                              fg=self.themes[self.theme.get()]["accent"])
        title_label.pack(side=tk.LEFT)
        
        theme_frame = tk.Frame(header_frame, bg=self.themes[self.theme.get()]["bg"])
        theme_frame.pack(side=tk.RIGHT)
        
        tk.Label(theme_frame, text="🎨 Theme:", font=('Segoe UI', 10), bg=self.themes[self.theme.get()]["bg"],
                fg=self.themes[self.theme.get()]["fg"]).pack(side=tk.LEFT, padx=(0, 5))
        
        theme_toggle = ttk.Combobox(theme_frame, textvariable=self.theme, values=["dark", "light"], 
                                   state="readonly", width=8)
        theme_toggle.pack(side=tk.LEFT)
        theme_toggle.bind('<<ComboboxSelected>>', lambda e: self.apply_theme())
        
        content_frame = tk.Frame(main_container, bg=self.themes[self.theme.get()]["bg"])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas_frame = tk.Frame(content_frame, bg=self.themes[self.theme.get()]["bg"])
        canvas_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Increase canvas width to match the desired panel width
        canvas = tk.Canvas(canvas_frame, bg=self.themes[self.theme.get()]["card_bg"], 
                          width=400, height=800,  # Increased width from 350 to 400
                          highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")  # Changed from right to left
        
        scrollable_frame = tk.Frame(canvas, bg=self.themes[self.theme.get()]["card_bg"])
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Make sure the scrollable frame uses the full canvas width
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_reqwidth())
        
        self.left_panel = scrollable_frame
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        mode_card = ttk.Frame(self.left_panel, style='Card.TFrame', padding=10)
        mode_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mode_card, text="⚙️ Signal Processing Mode", style='Title.TLabel').pack(anchor=tk.W)
        
        mode_frame = tk.Frame(mode_card, bg=self.themes[self.theme.get()]["card_bg"])
        mode_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.discrete_btn = tk.Radiobutton(mode_frame, text="🔢 Discrete-Time", variable=self.mode, value="discrete",
                                          command=self.on_mode_change, bg=self.themes[self.theme.get()]["card_bg"],
                                          fg=self.themes[self.theme.get()]["fg"], font=('Segoe UI', 10),
                                          activebackground=self.themes[self.theme.get()]["accent"],
                                          selectcolor=self.themes[self.theme.get()]["accent"])
        self.discrete_btn.pack(anchor=tk.W, pady=2)
        
        self.continuous_btn = tk.Radiobutton(mode_frame, text="📊 Continuous-Time", variable=self.mode, value="continuous",
                                           command=self.on_mode_change, bg=self.themes[self.theme.get()]["card_bg"],
                                           fg=self.themes[self.theme.get()]["fg"], font=('Segoe UI', 10),
                                           activebackground=self.themes[self.theme.get()]["accent"],
                                           selectcolor=self.themes[self.theme.get()]["accent"])
        self.continuous_btn.pack(anchor=tk.W, pady=2)
        
        self.input_card = ttk.Frame(self.left_panel, style='Card.TFrame', padding=10)
        self.input_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.input_card, text="📈 Input Signal Configuration", style='Title.TLabel').pack(anchor=tk.W)
        
        self.impulse_card = ttk.Frame(self.left_panel, style='Card.TFrame', padding=10)
        self.impulse_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.impulse_card, text="⚡ Impulse Response Setup", style='Title.TLabel').pack(anchor=tk.W)
        
        animation_card = ttk.Frame(self.left_panel, style='Card.TFrame', padding=10)
        animation_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(animation_card, text="🎬 Animation Controls", style='Title.TLabel').pack(anchor=tk.W)
        
        speed_frame = tk.Frame(animation_card, bg=self.themes[self.theme.get()]["card_bg"])
        speed_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(speed_frame, text="🚀 Speed:", style='Body.TLabel').pack(anchor=tk.W)
        self.speed_var = tk.IntVar(value=150)
        speed_scale = ttk.Scale(speed_frame, from_=5, to=1, variable=self.speed_var,
                               orient=tk.HORIZONTAL, command=self.update_speed)
        speed_scale.pack(fill=tk.X, pady=(5, 10))
        
        quality_frame = tk.Frame(animation_card, bg=self.themes[self.theme.get()]["card_bg"])
        quality_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(quality_frame, text="✨ Quality:", style='Body.TLabel').pack(anchor=tk.W)
        quality_combo = ttk.Combobox(quality_frame, textvariable=self.animation_quality,
                                    values=["standard", "high"], state="readonly")
        quality_combo.pack(fill=tk.X, pady=(5, 0))
        
        action_card = ttk.Frame(self.left_panel, style='Card.TFrame', padding=10)
        action_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(action_card, text="🎯 Actions", style='Title.TLabel').pack(anchor=tk.W)
        
        button_frame = tk.Frame(action_card, bg=self.themes[self.theme.get()]["card_bg"])
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.conv_btn = ttk.Button(button_frame, text="🔄 Compute Convolution",
                                  command=self.compute_convolution, style='Accent.TButton')
        self.conv_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.corr_btn = ttk.Button(button_frame, text="📊 Compute Correlation",
                                  command=self.compute_correlation, style='Secondary.TButton')
        self.corr_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.reset_btn = ttk.Button(button_frame, text="🔄 Reset All",
                                   command=self.reset_all, style='Success.TButton')
        self.reset_btn.pack(fill=tk.X)
        
        plot_panel = tk.Frame(content_frame, bg=self.themes[self.theme.get()]["bg"])
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        plot_header = tk.Frame(plot_panel, bg=self.themes[self.theme.get()]["bg"])
        plot_header.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(plot_header, text="📊 Real-time Signal Visualization", font=('Segoe UI', 16, 'bold'),
                bg=self.themes[self.theme.get()]["bg"], fg=self.themes[self.theme.get()]["accent"]).pack(anchor=tk.W)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.patch.set_facecolor(self.themes[self.theme.get()]["plot_bg"])
        self.fig.tight_layout(pad=4.0)
        
        for i, ax in enumerate([self.ax1, self.ax2, self.ax3]):
            ax.set_facecolor(self.themes[self.theme.get()]["plot_bg"])
            ax.spines['top'].set_color(self.themes[self.theme.get()]["accent"])
            ax.spines['bottom'].set_color(self.themes[self.theme.get()]["accent"])
            ax.spines['left'].set_color(self.themes[self.theme.get()]["accent"])
            ax.spines['right'].set_color(self.themes[self.theme.get()]["accent"])
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
        
        canvas_frame = tk.Frame(plot_panel, bg=self.themes[self.theme.get()]["bg"], relief='solid', bd=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.on_mode_change()
        
    def create_custom_entry(self, parent, placeholder="", default_value=""):
        frame = tk.Frame(parent, bg=self.themes[self.theme.get()]["card_bg"])
        frame.pack(fill=tk.X, pady=(2, 5))
        
        entry = tk.Entry(frame, bg=self.themes[self.theme.get()]["entry_bg"], fg=self.themes[self.theme.get()]["entry_fg"],
                        font=('Segoe UI', 10), relief='flat', bd=3, insertbackground=self.themes[self.theme.get()]["accent"])
        entry.pack(fill=tk.X, ipady=5)
        
        if default_value:
            entry.insert(0, default_value)
            
        return entry
        
    def create_custom_combobox(self, parent, values, default_value=""):
        frame = tk.Frame(parent, bg=self.themes[self.theme.get()]["card_bg"])
        frame.pack(fill=tk.X, pady=(2, 5))
        
        var = tk.StringVar(value=default_value)
        combo = ttk.Combobox(frame, textvariable=var, values=values, state="readonly", font=('Segoe UI', 10))
        combo.pack(fill=tk.X, ipady=5)
        
        return combo, var
        
    def on_mode_change(self):
        for widget in self.input_card.winfo_children()[1:]:
            widget.destroy()
        for widget in self.impulse_card.winfo_children()[1:]:
            widget.destroy()
            
        if self.mode.get() == "discrete":
            self.setup_discrete_inputs()
        else:
            self.setup_continuous_inputs()
            
    def setup_discrete_inputs(self):
        input_frame = tk.Frame(self.input_card, bg=self.themes[self.theme.get()]["card_bg"])
        input_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(input_frame, text="📊 Input Vector:", style='Heading.TLabel').pack(anchor=tk.W)
        self.input_vector_entry = self.create_custom_entry(input_frame, default_value="1,2,3,2,1")
        
        ttk.Label(input_frame, text="🎯 Start Index:", style='Heading.TLabel').pack(anchor=tk.W)
        self.input_start_entry = self.create_custom_entry(input_frame, default_value="-2")
        
        impulse_frame = tk.Frame(self.impulse_card, bg=self.themes[self.theme.get()]["card_bg"])
        impulse_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(impulse_frame, text="⚡ Impulse Vector:", style='Heading.TLabel').pack(anchor=tk.W)
        self.impulse_vector_entry = self.create_custom_entry(impulse_frame, default_value="1,0.8,0.6,0.4,0.2")
        
        ttk.Label(impulse_frame, text="🎯 Start Index:", style='Heading.TLabel').pack(anchor=tk.W)
        self.impulse_start_entry = self.create_custom_entry(impulse_frame, default_value="0")
        
    def setup_continuous_inputs(self):
        input_frame = tk.Frame(self.input_card, bg=self.themes[self.theme.get()]["card_bg"])
        input_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(input_frame, text="📊 Signal Type:", style='Heading.TLabel').pack(anchor=tk.W)
        self.input_type_combo, self.input_type_var = self.create_custom_combobox(
            input_frame, ["impulse", "step", "triangular", "rectangular", "sawtooth"], "rectangular")
        
        ttk.Label(input_frame, text="⏰ Start Time:", style='Heading.TLabel').pack(anchor=tk.W)
        self.input_start_time_entry = self.create_custom_entry(input_frame, default_value="0")
        
        ttk.Label(input_frame, text="⏰ End Time:", style='Heading.TLabel').pack(anchor=tk.W)
        self.input_end_time_entry = self.create_custom_entry(input_frame, default_value="3")
        
        ttk.Label(input_frame, text="📈 Amplitude:", style='Heading.TLabel').pack(anchor=tk.W)
        self.input_amplitude_entry = self.create_custom_entry(input_frame, default_value="1")
        
        impulse_frame = tk.Frame(self.impulse_card, bg=self.themes[self.theme.get()]["card_bg"])
        impulse_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(impulse_frame, text="⚡ Signal Type:", style='Heading.TLabel').pack(anchor=tk.W)
        self.impulse_type_combo, self.impulse_type_var = self.create_custom_combobox(
            impulse_frame, ["impulse", "step", "triangular", "rectangular", "sawtooth"], "triangular")
        
        ttk.Label(impulse_frame, text="⏰ Start Time:", style='Heading.TLabel').pack(anchor=tk.W)
        self.impulse_start_time_entry = self.create_custom_entry(impulse_frame, default_value="0")
        
        ttk.Label(impulse_frame, text="⏰ End Time:", style='Heading.TLabel').pack(anchor=tk.W)
        self.impulse_end_time_entry = self.create_custom_entry(impulse_frame, default_value="2")
        
        ttk.Label(impulse_frame, text="📈 Amplitude:", style='Heading.TLabel').pack(anchor=tk.W)
        self.impulse_amplitude_entry = self.create_custom_entry(impulse_frame, default_value="1")
        
    def update_speed(self, value):
        self.animation_speed = int(float(value))
        
    def parse_discrete_signals(self):
        try:
            input_vector = list(map(float, self.input_vector_entry.get().split(',')))
            input_start = int(self.input_start_entry.get())
            
            impulse_vector = list(map(float, self.impulse_vector_entry.get().split(',')))
            impulse_start = int(self.impulse_start_entry.get())
            
            self.input_signal = np.array(input_vector)
            self.input_indices = np.arange(input_start, input_start + len(input_vector))
            
            self.impulse_response = np.array(impulse_vector)
            self.impulse_indices = np.arange(impulse_start, impulse_start + len(impulse_vector))
            
            return True
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Invalid input: {str(e)}")
            return False
            
    def generate_continuous_signal(self, signal_type, start_time, end_time, amplitude, dt=0.02):
        t = np.arange(start_time, end_time, dt)
        
        if signal_type == "impulse":
            signal = np.zeros_like(t)
            if len(t) > 0:
                signal[0] = amplitude / dt
        elif signal_type == "step":
            signal = amplitude * np.ones_like(t)
        elif signal_type == "triangular":
            duration = end_time - start_time
            mid_point = start_time + duration / 2
            signal = amplitude * (1 - 2 * np.abs(t - mid_point) / duration)
        elif signal_type == "rectangular":
            signal = amplitude * np.ones_like(t)
        elif signal_type == "sawtooth":
            duration = end_time - start_time
            signal = amplitude * (t - start_time) / duration
            
        return t, signal
        
    def parse_continuous_signals(self):
        try:
            input_start = float(self.input_start_time_entry.get())
            input_end = float(self.input_end_time_entry.get())
            input_amp = float(self.input_amplitude_entry.get())
            
            impulse_start = float(self.impulse_start_time_entry.get())
            impulse_end = float(self.impulse_end_time_entry.get())
            impulse_amp = float(self.impulse_amplitude_entry.get())
            
            self.input_indices, self.input_signal = self.generate_continuous_signal(
                self.input_type_var.get(), input_start, input_end, input_amp)
            
            impulse_indices, impulse_signal = self.generate_continuous_signal(
                self.impulse_type_var.get(), impulse_start, impulse_end, impulse_amp)
            
            self.impulse_indices = impulse_indices
            self.impulse_response = impulse_signal
            
            return True
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Invalid input: {str(e)}")
            return False
            
    def compute_convolution(self):
        if self.mode.get() == "discrete":
            if not self.parse_discrete_signals():
                return
        else:
            if not self.parse_continuous_signals():
                return
                
        if self.mode.get() == "discrete":
            self.output_signal = np.convolve(self.input_signal, self.impulse_response, mode='full')
            self.output_indices = np.arange(
                self.input_indices[0] + self.impulse_indices[0],
                self.input_indices[0] + self.impulse_indices[0] + len(self.output_signal)
            )
        else:
            dt = self.input_indices[1] - self.input_indices[0]
            self.output_signal = np.convolve(self.input_signal, self.impulse_response, mode='full') * dt
            
            output_start = self.input_indices[0] + self.impulse_indices[0]
            output_end = output_start + len(self.output_signal) * dt
            self.output_indices = np.arange(output_start, output_end, dt)[:len(self.output_signal)]
            
        self.animate_convolution_enhanced()
        
    def compute_correlation(self):
        if self.mode.get() == "discrete":
            if not self.parse_discrete_signals():
                return
        else:
            if not self.parse_continuous_signals():
                return
                
        if self.mode.get() == "discrete":
            correlation = np.correlate(self.input_signal, self.impulse_response, mode='full')
            corr_indices = np.arange(
                self.input_indices[0] - self.impulse_indices[-1],
                self.input_indices[0] - self.impulse_indices[-1] + len(correlation)
            )
        else:
            dt = self.input_indices[1] - self.input_indices[0]
            correlation = np.correlate(self.input_signal, self.impulse_response, mode='full') * dt
            
            corr_start = self.input_indices[0] - self.impulse_indices[-1]
            corr_end = corr_start + len(correlation) * dt
            corr_indices = np.arange(corr_start, corr_end, dt)[:len(correlation)]
            
        self.ax3.clear()
        self.setup_enhanced_plot(self.ax3, "📊 Correlation Result")
        
        self.ax3.fill_between(corr_indices, correlation, alpha=0.3, color=self.themes[self.theme.get()]["success"])
        self.ax3.plot(corr_indices, correlation, color=self.themes[self.theme.get()]["success"], linewidth=3, label='Correlation')
        
        peaks = np.where(np.abs(correlation) > 0.7 * np.max(np.abs(correlation)))[0]
        if len(peaks) > 0:
            self.ax3.scatter(corr_indices[peaks], correlation[peaks], color=self.themes[self.theme.get()]["warning"], s=100, zorder=5)
        
        self.ax3.legend()
        self.canvas.draw()
        
    def setup_enhanced_plot(self, ax, title):
        current_theme = self.themes[self.theme.get()]
        ax.set_facecolor(current_theme["plot_bg"])
        ax.set_title(title, fontsize=14, fontweight='bold', color=current_theme["accent"])
        ax.set_xlabel('Time/Index', fontsize=12, color=current_theme["plot_fg"])
        ax.xaxis.set_label_coords(0.95, -0.099) 
        ax.set_ylabel('Amplitude', fontsize=12, color=current_theme["plot_fg"])
        ax.grid(True, alpha=0.3, color=current_theme["grid_color"])
        ax.spines['top'].set_color(current_theme["accent"])
        ax.spines['bottom'].set_color(current_theme["accent"])
        ax.spines['left'].set_color(current_theme["accent"])
        ax.spines['right'].set_color(current_theme["accent"])
        
    def animate_convolution_enhanced(self):
        self.animation_running = True
        self.current_frame = 0
        self.total_frames = len(self.output_signal)
        
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Setup enhanced plots
        self.setup_enhanced_plot(self.ax1, "Input Signal")
        self.setup_enhanced_plot(self.ax2, "⚡ Shifting Impulse Response")
        self.setup_enhanced_plot(self.ax3, "Convolution Output")
        
        # Enhanced input signal plot
        self.ax1.fill_between(self.input_indices, self.input_signal, alpha=0.3, 
                             color=self.themes[self.theme.get()]["accent"])
        self.ax1.plot(self.input_indices, self.input_signal, 
                     color=self.themes[self.theme.get()]["accent"], 
                     linewidth=4, marker='o', markersize=6, label='Input Signal')
        
        # Add glow effect for input signal
        if self.animation_quality.get() in ["high"]:
            for i in range(3):
                self.ax1.plot(self.input_indices, self.input_signal, 
                             color=self.themes[self.theme.get()]["accent"], 
                             linewidth=4-i, alpha=0.3-i*0.1)
        
        self.ax1.legend()
        
        # Start animation in separate thread
        self.animation_thread = threading.Thread(target=self.run_enhanced_animation)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def run_enhanced_animation(self):
        output_values = []
        glow_cycle = 0
        
        # First flip the impulse response for convolution
        flipped_impulse = self.impulse_response[::-1]
        
        for i in range(len(self.output_signal)):
            if not self.animation_running:
                break
                
            self.current_frame = i
            glow_cycle += 0.1
            self.glow_intensity = 0.5 + 0.5 * np.sin(glow_cycle)
            
            # Enhanced shifting impulse response
            self.ax2.clear()
            self.setup_enhanced_plot(self.ax2, f"⚡ Shifting Flipped Impulse Response (Frame {i+1}/{len(self.output_signal)})")
            
            if self.mode.get() == "discrete":
                # Start with the last point of flipped impulse aligned with first point of input
                initial_shift = self.input_indices[0] - (len(flipped_impulse) - 1)  # Align last point with first input point
                shift = initial_shift + i  # Move right to left
                shifted_impulse_indices = np.arange(len(flipped_impulse)) + shift
                shifted_impulse = flipped_impulse
            else:
                # For continuous signals
                dt = self.input_indices[1] - self.input_indices[0]
                initial_shift = self.input_indices[0] - (len(flipped_impulse) - 1) * dt
                shift = initial_shift + i * dt
                shifted_impulse_indices = np.arange(len(flipped_impulse)) * dt + shift
                shifted_impulse = flipped_impulse
            
            # Plot input signal as background
            self.ax2.plot(self.input_indices, self.input_signal, 
                        color=self.themes[self.theme.get()]["accent"], 
                        alpha=0.4, linewidth=2, linestyle='--', label='Input Signal')
            
            # Enhanced shifting impulse with glow
            self.ax2.fill_between(shifted_impulse_indices, shifted_impulse, alpha=0.3, 
                                color=self.themes[self.theme.get()]["secondary"])
            
            # Main impulse plot
            self.ax2.plot(shifted_impulse_indices, shifted_impulse, 
                        color=self.themes[self.theme.get()]["secondary"], 
                        linewidth=4, marker='s', markersize=8, 
                        label='Flipped & Shifted Impulse')
            
            # Add glow effect for high quality
            if self.animation_quality.get() in ["high"]:
                for j in range(3):
                    self.ax2.plot(shifted_impulse_indices, shifted_impulse, 
                                color=self.themes[self.theme.get()]["secondary"], 
                                linewidth=6-j*2, alpha=0.3*self.glow_intensity)
            
            # Highlight overlap region
            if self.mode.get() == "discrete":
                overlap_start = max(self.input_indices[0], shifted_impulse_indices[0])
                overlap_end = min(self.input_indices[-1], shifted_impulse_indices[-1])
                if overlap_start <= overlap_end:
                    overlap_x = [overlap_start, overlap_end]
                    overlap_y = [0, 0]
                    self.ax2.fill_between(overlap_x, [-1, -1], [1, 1], 
                                        alpha=0.2, color=self.themes[self.theme.get()]["warning"])
            
            self.ax2.legend()
            #Enhanced output signal animation
            output_values.append(self.output_signal[i])
            self.ax3.clear()
            self.setup_enhanced_plot(self.ax3, f"Convolution Output (Building... {i+1}/{len(self.output_signal)})")
            
            # Gradient fill for output
            if len(output_values) > 1:
                self.ax3.fill_between(self.output_indices[:i+1], output_values, alpha=0.3, 
                                     color=self.themes[self.theme.get()]["success"])
            
            # Main output plot with enhanced styling
            self.ax3.plot(self.output_indices[:i+1], output_values, 
                         color=self.themes[self.theme.get()]["success"], 
                         linewidth=4, marker='o', markersize=8, 
                         markerfacecolor=self.themes[self.theme.get()]["warning"],
                         markeredgecolor=self.themes[self.theme.get()]["success"],
                         markeredgewidth=2)
            
            # Add glow effect for current point
            if self.animation_quality.get() in ["high"]:
                current_point = self.output_indices[i]
                current_value = output_values[i]
                
                # Pulsing glow effect
                for j in range(5):
                    self.ax3.scatter(current_point, current_value, 
                                   s=200-j*30, alpha=0.3*self.glow_intensity, 
                                   color=self.themes[self.theme.get()]["warning"])
            
            # Add trailing effect for ultra quality
            if self.animation_quality.get() == "ultra" and i > 10:
                trail_indices = self.output_indices[max(0, i-10):i+1]
                trail_values = output_values[max(0, i-10):i+1]
                trail_alphas = np.linspace(0.1, 1.0, len(trail_values))
                
                for j in range(len(trail_values)):
                    self.ax3.scatter(trail_indices[j], trail_values[j], 
                                   s=50, alpha=trail_alphas[j]*0.5, 
                                   color=self.themes[self.theme.get()]["success"])
            
            # Progress bar effect
            progress = (i + 1) / len(self.output_signal)
            self.ax3.text(0.02, 0.98, f"Progress: {progress:.1%}", 
                         transform=self.ax3.transAxes, fontsize=10, 
                         verticalalignment='top', 
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor=self.themes[self.theme.get()]["card_bg"],
                                  alpha=0.8))
            
            # Update canvas
            self.canvas.draw()
            
            # Dynamic speed based on animation quality
            quality_speeds = {"standard": 1.2, "high": 0.5}
            adjusted_speed = self.animation_speed / quality_speeds[self.animation_quality.get()]
            time.sleep(adjusted_speed / 1000.0)
            
        # Final enhanced plot
        if self.animation_running:
            self.ax3.clear()
            self.setup_enhanced_plot(self.ax3, "Final Convolution Result")
            
            # Enhanced final plot with multiple visual elements
            self.ax3.fill_between(self.output_indices, self.output_signal, alpha=0.3, 
                                 color=self.themes[self.theme.get()]["success"])
            self.ax3.plot(self.output_indices, self.output_signal, 
                         color=self.themes[self.theme.get()]["success"], 
                         linewidth=4, marker='o', markersize=6)
            
            # Add statistical annotations
            max_val = np.max(self.output_signal)
            max_idx = np.argmax(self.output_signal)
            max_time = self.output_indices[max_idx]
            
            self.ax3.annotate(f'Max: {max_val:.2f}', 
                            xy=(max_time, max_val), xytext=(max_time, max_val + (max_val*-0.2)),
                            arrowprops=dict(arrowstyle='->', 
                                          color=self.themes[self.theme.get()]["warning"],
                                          lw=2),
                            fontsize=12, fontweight='bold',
                            color=self.themes[self.theme.get()]["warning"])
            
            # Add glow effect for final result
            if self.animation_quality.get() in ["high"]:
                for j in range(3):
                    self.ax3.plot(self.output_indices, self.output_signal, 
                                 color=self.themes[self.theme.get()]["success"], 
                                 linewidth=6-j*2, alpha=0.3-j*0.1)
            
            self.canvas.draw()
            
        self.animation_running = False

        

    def reset_all(self):
        self.animation_running = False
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            self.setup_enhanced_plot(ax, "Ready for New Signals")
            
        self.canvas.draw()
        self.input_signal = None
        self.impulse_response = None
        self.output_signal = None
        self.current_frame = 0
        self.total_frames = 0
        self.particle_effects = []

def main():
    root = tk.Tk()
    root.iconbitmap = lambda x: None 
    root.resizable(True, True)
    root.minsize(1200, 800)

    app = ConvolutionGUI(root)
    
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1400 // 2)
    y = (root.winfo_screenheight() // 2) - (900 // 2)
    root.geometry(f"1400x900+{x}+{y}")
    root.mainloop()

if __name__ == "__main__":
    main()