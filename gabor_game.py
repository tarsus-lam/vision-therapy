import tkinter as tk
from tkinter import ttk
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import product

# Gabor Kernel Function
def gabor_kernel(ksize, sigma, theta, lambd, gamma, psi, resolution, contrast):
    # Normalize contrast
    contrast = contrast * 3.0 / 5.0

    # Adjust sigma based on the contrast
    sigma_x = sigma * contrast
    sigma_y = (sigma / gamma) * contrast

    # Bounding box
    xmax = ksize // 2
    ymax = ksize // 2
    xmin = -xmax
    ymin = -ymax

    # Create a higher-resolution grid
    (y, x) = np.meshgrid(
        np.linspace(ymin, ymax, int((ymax - ymin + 1) * resolution)),
        np.linspace(xmin, xmax, int((xmax - xmin + 1) * resolution))
    )

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor kernel calculation
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / lambd * x_theta + psi)
    return gb


# App Class
class GaborGameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gabor Matching Game")

        # Detect screen resolution
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Scale the window size
        self.root.geometry(f"{screen_width}x{screen_height}")

        # Calculate scale factors
        self.scale_factor = min(screen_width / 1920, screen_height / 1080)

        # Scaled sizes
        self.title_font_size = int(30 * self.scale_factor)
        self.label_font_size = int(18 * self.scale_factor)
        self.button_width = int(12 * self.scale_factor)
        self.button_height = int(6 * self.scale_factor)
        self.patch_figsize = 2 * self.scale_factor

        # Initialize Variables
        self.color = tk.StringVar(value='Grays')
        self.grid_size = tk.StringVar(value='5x4')
        self.difficulty = tk.StringVar(value='Assorted')
        self.score = 0
        self.start_time = 0
        self.selected_cells = []
        self.grid_buttons = []

        # Create Start Screen
        self.create_start_screen()

    def create_start_screen(self):
        def update_grid_state(*args):
            if self.difficulty.get() == "Easy":
                self.grid_size.set("5x4")
                grid_dropdown.config(state="disabled")
            else:
                grid_dropdown.config(state="normal")

        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Use a single frame to manage layout consistency
        start_frame = tk.Frame(self.root)
        start_frame.pack()

        tk.Label(start_frame, text="Gabor Matching Game", font=("Arial", 20)).grid(row=0, column=0, columnspan=2, pady=10)

        # Color Dropdown
        tk.Label(start_frame, text="Choose Color:").grid(row=1, column=0, pady=5)
        color_dropdown = ttk.Combobox(start_frame, textvariable=self.color)
        color_dropdown['values'] = ['Grays', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
        color_dropdown.grid(row=1, column=1, pady=5)

        # Grid Size Dropdown
        tk.Label(start_frame, text="Choose Grid Size:").grid(row=2, column=0, pady=5)
        grid_dropdown = ttk.Combobox(start_frame, textvariable=self.grid_size)
        grid_dropdown['values'] = ['5x4', '5x6', '5x8']
        grid_dropdown.grid(row=2, column=1, pady=5)

        # Difficulty Dropdown
        tk.Label(start_frame, text="Choose Difficulty:").grid(row=3, column=0, pady=5)
        difficulty_dropdown = ttk.Combobox(start_frame, textvariable=self.difficulty)
        difficulty_dropdown['values'] = ['Assorted', 'Easy', 'Intermediate', 'Hard']
        difficulty_dropdown.grid(row=3, column=1, pady=5)

        # Apply initial state based on previous settings
        update_grid_state()

        # Trace difficulty to update grid size behavior
        self.difficulty.trace_add("write", update_grid_state)

        # Start Button
        start_button = tk.Button(start_frame, text="Start Game", command=self.start_game)
        start_button.grid(row=4, column=0, columnspan=2, pady=20)

    def start_game(self):
        # Reset variables
        self.score = 0
        self.start_time = time.time()
        self.selected_cells = []

        # Get settings
        color = self.color.get()
        difficulty = self.difficulty.get()

        # Force grid size to 5x4 if difficulty is "Easy"
        if difficulty == "Easy":
            self.grid_size.set("5x4")

        grid_size = self.grid_size.get().split('x')

        # Generate Gabor patches
        self.gabor_grid = self.generate_gabor_grid(grid_size, difficulty)

        # Create Game Screen
        self.create_game_screen()

    def generate_gabor_grid(self, grid_size, difficulty):
        # Set difficulty-specific parameters
        params = {
            'Assorted': {
                'theta': np.linspace(-np.pi, 3*np.pi/4, 8),
                'lambd': list(range(8, 21, 4)),
                'psi': np.linspace(-np.pi/2, np.pi, 4),
                'contrast': list(range(1, 6))
            },
            'Easy': {
                'theta': [np.pi, np.pi / 2],
                'lambd': list(range(12, 21, 4)),
                'psi': [0, np.pi],
                'contrast': [4, 5]
            },
            'Intermediate': {
                'theta': [np.pi, np.pi / 2, 3 * np.pi / 4, np.pi / 4],
                'lambd': list(range(8, 13, 2)),
                'psi': [0, np.pi],
                'contrast': [2, 3]
            },
            'Hard': {
                'theta': np.linspace(-np.pi, 3*np.pi/4, 8),
                'lambd': list(range(8, 13, 2)),
                'psi': np.linspace(-np.pi/2, np.pi, 4),
                'contrast': [1]
            }
        }

        selected_params = params[difficulty]

        # Generate all unique combinations of theta, lambda, psi
        all_combinations = list(product(
            selected_params['theta'], 
            selected_params['lambd'], 
            selected_params['psi']
        ))

        np.random.shuffle(all_combinations)  # Shuffle to ensure randomness

        # Calculate required number of patches
        grid_rows, grid_cols = map(int, grid_size)
        num_patches = grid_rows * grid_cols // 2

        # If there are not enough unique combinations, repeat some
        if len(all_combinations) < num_patches:
            all_combinations *= (num_patches // len(all_combinations)) + 1

        # Select unique combinations for the required number of patches
        selected_combinations = all_combinations[:num_patches]

        # Generate Gabor kernels using the unique combinations
        patches = []
        for theta, lambd, psi in selected_combinations:
            contrast = np.random.choice(selected_params['contrast'])
            kernel = gabor_kernel(31, 10, theta, lambd, 0.5, psi, 10, contrast)
            patches.append(kernel)

        # Duplicate pairs and shuffle
        patches *= 2  # Duplicate pairs
        np.random.shuffle(patches)
        return patches

    def create_game_screen(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Match the Gabor Patches!", font=("Arial", 20), fg="white").pack(pady=5)

        # Use a single frame for layout consistency
        game_frame = tk.Frame(self.root)
        game_frame.pack(fill="both", expand=True)
        game_frame.grid_columnconfigure(0, weight=1)
        game_frame.grid_columnconfigure(1, weight=1)
        game_frame.grid_columnconfigure(2, weight=1)
        game_frame.grid_columnconfigure(3, weight=1)

        # Score and Timer Display
        self.score_label = tk.Label(game_frame, text=f"Score: {self.score}", font=("Arial", 28), fg="white")
        self.score_label.grid(row=0, column=0, columnspan=3, pady=5)
        self.timer_label = tk.Label(game_frame, text="Time: 0s", font=("Arial", 28), fg="white")
        self.timer_label.grid(row=0, column=2, columnspan=1, pady=5)

        # Start Timer
        self.update_timer()

        # Display grid of Gabor patches
        grid_frame = tk.Frame(game_frame)
        grid_frame.grid(row=1, column=0, columnspan=4, pady=15)

        # Configure uniform row/column sizes
        grid_rows, grid_cols = map(int, self.grid_size.get().split('x'))
        
        for r in range(grid_rows):
            grid_frame.rowconfigure(r, weight=1)
        for c in range(grid_cols):
            grid_frame.columnconfigure(c, weight=1)

        self.grid_buttons = []
        self.highlights = {}  # Store highlight rectangles

        for i, kernel in enumerate(self.gabor_grid):
            # Create a container frame
            row, col = divmod(i, grid_cols)
            patch_frame = tk.Frame(grid_frame)
            patch_frame.grid(row=row, column=col, padx=4, pady=4)

            # Create a canvas to display the Gabor patch
            canvas = tk.Canvas(patch_frame, width=100, height=100, highlightthickness=0)
            canvas.pack(fill="both", expand=True)

            # Draw the Gabor patch
            fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
            ax.imshow(kernel, cmap=self.color.get(), extent=(-15, 15, -15, 15))
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Render the figure to an image and add it to the canvas
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            # Convert the image to PhotoImage
            from PIL import Image, ImageTk
            img = Image.fromarray(img_array)
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(50, 50, image=photo)
            canvas.image = photo  # Store reference to prevent garbage collection

            # Bind click event
            canvas.bind("<Button-1>", lambda event, idx=i: self.on_patch_click(idx))

            # Store the canvas and frame
            self.grid_buttons.append(canvas)

        tk.Button(self.root, text="End Game", command=self.end_game).pack(pady=30)

    def on_patch_click(self, idx):
        if idx in self.selected_cells or idx in self.highlights:
            return

        self.selected_cells.append(idx)

        # Highlight the clicked patch
        canvas = self.grid_buttons[idx]
        if idx not in self.highlights:
            rect = canvas.create_rectangle(0, 0, 100, 100, outline="yellow", width=4)
            self.highlights[idx] = rect

        if len(self.selected_cells) == 2:
            self.check_match()

    def check_match(self):
        idx1, idx2 = self.selected_cells
        if np.allclose(self.gabor_grid[idx1], self.gabor_grid[idx2]):
            self.score += 5

            # Keep highlights as permanent
            self.highlights[idx1] = self.highlights[idx1]
            self.highlights[idx2] = self.highlights[idx2]

            # Check if all cells are matched
            if len(self.highlights) == len(self.gabor_grid):
                self.end_game()
        else:
            self.score -= 5
            self.root.after(500, lambda: self.reset_incorrect(idx1, idx2))
        self.score_label.config(text=f"Score: {self.score}")
        self.selected_cells = []


    def reset_incorrect(self, idx1, idx2):
        self.grid_buttons[idx1].delete(self.highlights[idx1])
        del self.highlights[idx1]
        
        self.grid_buttons[idx2].delete(self.highlights[idx2])
        del self.highlights[idx2]

    def update_timer(self):
        elapsed_time = int(time.time() - self.start_time)
        self.timer_label.config(text=f"Time: {elapsed_time}s")
        self.root.after(1000, self.update_timer)

    def end_game(self):
        elapsed_time = int(time.time() - self.start_time)
        self.create_end_screen(elapsed_time)

    def create_end_screen(self, elapsed_time):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        end_frame = tk.Frame(self.root)
        end_frame.pack()

        tk.Label(end_frame, text=f"Final Score: {self.score}", font=("Arial", 28)).pack(pady=5)
        tk.Label(end_frame, text=f"Time Taken: {elapsed_time}s", font=("Arial", 28)).pack(pady=5)
        tk.Button(end_frame, text="Play Again", command=self.create_start_screen).pack(pady=10)


# Main
root = tk.Tk()
app = GaborGameApp(root)
root.mainloop()

