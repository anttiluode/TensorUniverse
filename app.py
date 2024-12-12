import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vispy
import gradio as gr
from scipy.spatial.transform import Rotation

class PhysicalTensorSingularity:
    def __init__(self, dimension=128, position=None, mass=1.0):
        self.dimension = dimension
        # Physical properties
        self.position = position if position is not None else np.random.rand(3)
        self.velocity = np.random.randn(3) * 0.1
        self.mass = mass
        # Tensor properties
        self.core = torch.randn(dimension)
        self.field = self.generate_gravitational_field()
        
    def generate_gravitational_field(self):
        """Generate gravitational field based on mass"""
        field = self.core.clone()
        # Apply gravitational influence
        r = torch.linspace(0, 2*np.pi, self.dimension)
        field *= torch.exp(-r/self.mass)  # Gravitational falloff
        return field
    
    def update_position(self, dt, force):
        """Update position using Newtonian physics"""
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class PhysicalTensorUniverse:
    def __init__(self, size=50, num_singularities=100, dimension=128):
        self.G = 6.67430e-11  # Gravitational constant
        self.size = size
        self.dimension = dimension
        self.space = np.zeros((size, size, size))
        self.singularities = []
        self.initialize_singularities(num_singularities)
        
    def initialize_singularities(self, num):
        """Initialize singularities with random positions and masses"""
        for _ in range(num):
            position = np.random.rand(3) * self.size
            mass = np.random.exponential(1.0)  # Random masses
            self.singularities.append(
                PhysicalTensorSingularity(
                    dimension=self.dimension,
                    position=position,
                    mass=mass
                )
            )
    
    def calculate_gravity(self, pos1, pos2, m1, m2):
        """Calculate gravitational force between two points"""
        r = pos2 - pos1
        distance = np.linalg.norm(r)
        if distance < 1e-10:
            return np.zeros(3)
        force_magnitude = self.G * m1 * m2 / (distance**2)
        return force_magnitude * r / distance
    
    def update_tensor_interactions(self):
        """Update tensor field interactions"""
        for s1 in self.singularities:
            force = np.zeros(3)
            for s2 in self.singularities:
                if s1 != s2:
                    # Calculate gravitational force
                    force += self.calculate_gravity(
                        s1.position, s2.position,
                        s1.mass, s2.mass
                    )
                    # Tensor field interaction
                    field_interaction = torch.tanh(s1.field * s2.field)
                    # Modify gravitational force based on field interaction
                    force *= 1 + torch.mean(field_interaction).item()
            s1.update_position(0.1, force)  # dt = 0.1
    
    def update_space(self):
        """Update 3D space based on singularity positions and fields"""
        self.space.fill(0)
        x = np.linspace(0, self.size, self.size)
        y = np.linspace(0, self.size, self.size)
        z = np.linspace(0, self.size, self.size)
        X, Y, Z = np.meshgrid(x, y, z)
        
        for s in self.singularities:
            # Calculate distance from singularity to each point
            R = np.sqrt((X - s.position[0])**2 + 
                       (Y - s.position[1])**2 + 
                       (Z - s.position[2])**2)
            # Add field influence
            self.space += s.mass / (R + 1) * np.mean(s.field.numpy())
    
    def detect_structures(self):
        """Detect galaxy-like structures"""
        structures = []
        density_threshold = np.mean(self.space) + np.std(self.space)
        
        # Find high density regions
        dense_regions = self.space > density_threshold
        
        # Basic clustering
        from scipy.ndimage import label
        labeled_array, num_features = label(dense_regions)
        
        for i in range(1, num_features + 1):
            structure = {
                'center': np.mean(np.where(labeled_array == i), axis=1),
                'mass': np.sum(self.space[labeled_array == i]),
                'size': np.sum(labeled_array == i)
            }
            structures.append(structure)
        
        return structures

def create_visualization(universe, step):
    """Create 3D visualization of the universe"""
    fig = plt.figure(figsize=(20, 15))
    
    # 3D density plot
    ax1 = fig.add_subplot(221, projection='3d')
    x, y, z = np.where(universe.space > np.mean(universe.space))
    scatter = ax1.scatter(x, y, z, c=universe.space[x,y,z], cmap='viridis')
    ax1.set_title('Matter Distribution')
    plt.colorbar(scatter, ax=ax1)
    
    # Structure detection
    structures = universe.detect_structures()
    if structures:
        # Plot structure centers
        centers = np.array([s['center'] for s in structures])
        sizes = np.array([s['size'] for s in structures])
        ax1.scatter(centers[:,0], centers[:,1], centers[:,2], 
                   c='red', s=sizes/10, alpha=0.5)
    
    # Density slice
    ax2 = fig.add_subplot(222)
    middle_slice = universe.space[:,:,universe.size//2]
    im = ax2.imshow(middle_slice, cmap='viridis')
    ax2.set_title('Density Slice')
    plt.colorbar(im, ax=ax2)
    
    # Field strength distribution
    ax3 = fig.add_subplot(223)
    field_strengths = [torch.mean(s.field).item() for s in universe.singularities]
    ax3.hist(field_strengths, bins=30)
    ax3.set_title('Field Strength Distribution')
    
    # Mass distribution
    ax4 = fig.add_subplot(224)
    masses = [s.mass for s in universe.singularities]
    ax4.hist(masses, bins=30)
    ax4.set_title('Mass Distribution')
    
    plt.tight_layout()
    return fig

def simulate_with_gradio():
    """Create Gradio interface for the simulation"""
    def run_simulation(num_singularities, size, steps):
        universe = PhysicalTensorUniverse(
            size=size,
            num_singularities=num_singularities
        )
        figures = []
        for step in range(steps):
            universe.update_tensor_interactions()
            universe.update_space()
            if step % 10 == 0:
                fig = create_visualization(universe, step)
                figures.append(fig)
        return figures
    
    interface = gr.Interface(
        fn=run_simulation,
        inputs=[
            gr.Slider(2, 1000, 100, label="Number of Singularities"),
            gr.Slider(10, 100, 50, label="Universe Size"),
            gr.Slider(10, 1000, 100, label="Simulation Steps")
        ],
        outputs=[gr.Plot() for _ in range(10)],
        title="Physical Tensor Universe Simulation",
        description="Simulate a universe with tensor singularities following physical laws"
    )
    
    return interface

if __name__ == "__main__":
    interface = simulate_with_gradio()
    interface.launch()