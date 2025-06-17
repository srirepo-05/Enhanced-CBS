#!/usr/bin/env python3
import pygame
import numpy as np
import math
import sys

# Colors
COLORS = [(0, 255, 0), (0, 0, 255), (255, 165, 0)]  # green, blue, orange
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIGHT_GRAY = (200, 200, 200)

class Animation:
    def __init__(self, my_map, starts, goals, paths, cell_size=20, fps=60):
        pygame.init()
        
        # Transform coordinates like in original code
        self.my_map = np.flip(np.transpose(my_map), 1)
        self.starts = []
        for start in starts:
            self.starts.append((start[1], len(self.my_map[0]) - 1 - start[0]))
        self.goals = []
        for goal in goals:
            self.goals.append((goal[1], len(self.my_map[0]) - 1 - goal[0]))
        self.paths = []
        if paths:
            for path in paths:
                self.paths.append([])
                for loc in path:
                    self.paths[-1].append((loc[1], len(self.my_map[0]) - 1 - loc[0]))
        
        # Display settings
        self.cell_size = cell_size
        self.fps = fps
        self.width = len(self.my_map) * cell_size
        self.height = len(self.my_map[0]) * cell_size
        
        # Initialize pygame
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maze Path Visualization")
        self.clock = pygame.time.Clock()
        
        # Animation variables
        self.t = 0.0
        self.max_t = 0
        if self.paths:
            self.max_t = max(len(path) - 1 for path in self.paths)
        
        # Agent properties
        self.agent_radius = int(cell_size * 0.3)
        self.goal_size = int(cell_size * 0.5)
        
        # Font for agent labels
        self.font = pygame.font.Font(None, int(cell_size * 0.4))
        
        self.running = True
        self.paused = False
        self.speed_multiplier = 0.3
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        screen_x = int((x + 0.5) * self.cell_size)
        screen_y = int((y + 0.5) * self.cell_size)
        return screen_x, screen_y
    
    def draw_map(self):
        """Draw the static map elements"""
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw boundary
        pygame.draw.rect(self.screen, GRAY, (0, 0, self.width, self.height), 2)
        
        # Draw obstacles
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[0])):
                if self.my_map[i][j]:
                    x = i * self.cell_size
                    y = j * self.cell_size
                    pygame.draw.rect(self.screen, GRAY, (x, y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, BLACK, (x, y, self.cell_size, self.cell_size), 1)
    
    def draw_goals(self):
        """Draw goal positions"""
        for i, goal in enumerate(self.goals):
            x, y = self.world_to_screen(goal[0], goal[1])
            color = COLORS[i % len(COLORS)]
            # Draw semi-transparent goal
            goal_rect = pygame.Rect(x - self.goal_size//2, y - self.goal_size//2, 
                                  self.goal_size, self.goal_size)
            
            # Create a surface for alpha blending
            goal_surface = pygame.Surface((self.goal_size, self.goal_size))
            goal_surface.set_alpha(128)  # 50% transparency
            goal_surface.fill(color)
            self.screen.blit(goal_surface, goal_rect)
            
            # Draw border
            pygame.draw.rect(self.screen, BLACK, goal_rect, 2)
    
    def get_state(self, t, path):
        """Get interpolated position at time t"""
        if t <= 0:
            return np.array(path[0])
        elif t >= len(path) - 1:
            return np.array(path[-1])
        else:
            idx = int(t)
            frac = t - idx
            pos_last = np.array(path[idx])
            pos_next = np.array(path[idx + 1])
            pos = (pos_next - pos_last) * frac + pos_last
            return pos
    
    def check_collisions(self, agent_positions):
        """Check for agent-agent collisions"""
        collisions = set()
        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                pos1 = np.array(agent_positions[i])
                pos2 = np.array(agent_positions[j])
                distance = np.linalg.norm(pos1 - pos2)
                if distance < 0.7:  # Same threshold as original
                    collisions.add(i)
                    collisions.add(j)
                    print(f"COLLISION! (agent-agent) ({i}, {j}) at time {self.t:.2f}")
        return collisions
    
    def draw_agents(self):
        """Draw agents at current positions"""
        if not self.paths:
            return
        
        agent_positions = []
        for i, path in enumerate(self.paths):
            pos = self.get_state(self.t, path)
            agent_positions.append(pos)
        
        # Check for collisions
        collisions = self.check_collisions(agent_positions)
        
        # Draw agents
        for i, pos in enumerate(agent_positions):
            x, y = self.world_to_screen(pos[0], pos[1])
            
            # Choose color (red if collision, normal color otherwise)
            color = RED if i in collisions else COLORS[i % len(COLORS)]
            
            # Draw agent circle
            pygame.draw.circle(self.screen, color, (x, y), self.agent_radius)
            pygame.draw.circle(self.screen, BLACK, (x, y), self.agent_radius, 2)
            
            # Draw agent label
            label = self.font.render(str(i), True, BLACK)
            label_rect = label.get_rect(center=(x, y - self.agent_radius - 10))
            self.screen.blit(label, label_rect)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Create UI surface
        ui_height = 30
        ui_surface = pygame.Surface((self.width, ui_height))
        ui_surface.set_alpha(200)
        ui_surface.fill(LIGHT_GRAY)
        
        # Draw time and controls info
        time_text = f"Time: {self.t:.2f}/{self.max_t} | Speed: {self.speed_multiplier:.1f}x"
        controls_text = "SPACE: Pause | ↑↓: Speed | R: Reset | Q: Quit"
        
        time_surface = self.font.render(time_text, True, BLACK)
        controls_surface = self.font.render(controls_text, True, BLACK)
        
        ui_surface.blit(time_surface, (10, 5))
        ui_surface.blit(controls_surface, (self.width - controls_surface.get_width() - 10, 5))
        
        self.screen.blit(ui_surface, (0, self.height - ui_height))
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.t = 0.0
                elif event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_UP:
                    self.speed_multiplier = min(5.0, self.speed_multiplier + 0.5)
                elif event.key == pygame.K_DOWN:
                    self.speed_multiplier = max(0.1, self.speed_multiplier - 0.5)
    
    def update(self, dt):
        """Update animation state"""
        if not self.paused and self.paths:
            # Increase time with speed multiplier
            # dt is in seconds, convert to animation time units
            self.t += dt * self.speed_multiplier * 10  # 10x faster than original
            
            # Loop animation
            if self.t > self.max_t + 1:
                self.t = 0.0
    
    def run(self):
        """Main animation loop"""
        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0  # Convert to seconds
            
            self.handle_events()
            self.update(dt)
            
            # Draw everything
            self.draw_map()
            self.draw_goals()
            self.draw_agents()
            self.draw_ui()
            
            pygame.display.flip()
        
        pygame.quit()
    
    def show(self):
        """Match the original interface - runs the pygame animation"""
        self.run()
    
    @staticmethod
    def show_static():
        """Static method to match original interface"""
        pass
    
    def save_frames(self, filename_prefix, duration=None):
        """Save animation frames as images"""
        if duration is None:
            duration = self.max_t + 1
        
        frame_count = int(duration * self.fps)
        saved_t = self.t
        
        for frame in range(frame_count):
            self.t = (frame / self.fps) * 10  # Convert back to time units
            
            self.draw_map()
            self.draw_goals() 
            self.draw_agents()
            
            filename = f"{filename_prefix}_{frame:04d}.png"
            pygame.image.save(self.screen, filename)
            print(f"Saved frame {frame + 1}/{frame_count}")
        
        self.t = saved_t
        print(f"Saved {frame_count} frames")



