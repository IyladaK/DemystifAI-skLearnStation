import pygame
import numpy as np
import matplotlib.pyplot as plt


def drawingWindow():
    # --- Config ---
    GRID_SIZE = 32
    COMPRESSED_SIZE = GRID_SIZE // 8
    CELL_SIZE = 200//GRID_SIZE
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE
    LINE_COLOR = (200, 200, 200)  # grid line color
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # --- Setup ---
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("8x8 Pixel Art")

    # Create 8x8 grid of 0s (0 = white, 1 = black)
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    def draw_grid():
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                color = WHITE if grid[r][c] else BLACK
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

    def get_cell(pos):
        x, y = pos
        col = x // CELL_SIZE
        row = y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return row, col
        return None, None

    def circlePoints(radius):
        pointArr = []
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x**2 + y**2 <= radius**2:
                    pointArr.append((x + row, y + col))
        return pointArr

    def ballpoint(value, radius):
        pointArr = circlePoints(radius)
        for pair in pointArr:
            grid[pair[0]][pair[1]] = value


    # --- Main loop ---
    running = True
    drawMode = True
    pressed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif (event.type == pygame.MOUSEBUTTONDOWN
                  or (event.type == pygame.MOUSEMOTION and pressed == True)):
                if 0 < (event.pos[0] and event.pos[1]) <  WINDOW_SIZE:
                    row, col = get_cell(event.pos)
                    pressed = True
                    if drawMode:
                        ballpoint(1, 2)
                    else:
                        ballpoint(0)

            elif event.type == pygame.MOUSEBUTTONUP:
                pressed = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # press C to clear
                    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                if event.key == pygame.K_m:
                    drawMode = not drawMode
                if event.key == pygame.K_KP_ENTER:
                    running = False

        screen.fill(BLACK)
        draw_grid()
        pygame.display.flip()

    pygame.quit()

    compArr = []
    max = 0
    for y in range(8):
        arr = []
        for x in range(8):
            sum = 0
            for i in range(x*COMPRESSED_SIZE, (x+1)*COMPRESSED_SIZE):
                for j in range(y*COMPRESSED_SIZE, (y+1)*COMPRESSED_SIZE):
                    sum += grid[j][i]
            if sum > max:
                max = sum
            arr.append(sum)
        compArr.append(arr)

    out = []
    for arr in compArr:
        for i in range(len(arr)):
            arr[i] = (arr[i]/max)*16 # normalize to 16
        out.append(arr)

    input_data = np.array(out).reshape(1, -1)  # shape (1, 64)

    plt.imshow(out, cmap='gray', vmin=0, vmax=16)
    return input_data, out

