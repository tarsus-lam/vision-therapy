# vision-therapy

## Gabor Matching Game
A Python desktop application that generates and displays Gabor patches in a grid-based matching game.
The project demonstrates how to procedurally generate parametric visual stimuli and embed them in an interactive GUI using Tkinter, NumPy, and Matplotlib.

⚠️ **This project is experimental and educational. It is not a medical device and does not replace professional eye care.**

### Features
- Procedural generation of Gabor patches
- Multiple difficulty levels with different stimulus parameter ranges
- Configurable grid sizes
- Full-screen, resolution-adaptive UI
- Memory-style matching gameplay
- Real-time score and timer tracking
- Colormap selection for visual customization

### Requirements
- Python 3.8+
- Python packages
`pip install numpy matplotlib opencv-python pillow`
(tkinter is included with most standard Python installations.)

### Running the Game
From the project directory:
`python gabor_game.py`
The application launches in full-screen mode and begins at the start menu.

### Application Flow
1. Start Screen
  - Select color map
  - Select grid size
  - Select difficulty
  - Start the game
2. Game Screen
  - Gabor patches are displayed in a grid
  - Click two patches to attempt a match
  - Matches increase score; mismatches decrease score
  - Timer runs continuously
3. End Screen
  - Displays final score and total time
  - Option to restart the game
