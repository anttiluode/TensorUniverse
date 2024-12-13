import numpy as np
import torch
import vispy
from vispy import scene
from vispy.color import get_colormap
import gradio as gr
from scipy.ndimage import label
from PIL import Image
import io
import tempfile

class PhysicalTensorSingularity:
    def __init__(self, dimension=128, position=None, mass=1.0, device='cpu'):
        self.dimension = dimension
        self.device = device
        # Physical properties
        if position is not None:
            if isinstance(position, np.ndarray):
                self.position = torch.from_numpy(position).float().to(self.device)
            else:
                # If position is already a tensor, clone and detach it
                self.position = position.clone().detach().float().to(self.device)
        else:
            self.position = torch.tensor(np.random.rand(3), dtype=torch.float32, device=self.device)
        self.velocity = torch.randn(3, device=self.device) * 0.1
        self.mass = mass
        # Tensor properties
        self.core = torch.randn(dimension, device=self.device)
        self.field = self.generate_gravitational_field()

    def generate_gravitational_field(self):
        """Generate gravitational field based on mass"""
        field = self.core.clone()
        # Apply gravitational influence
        r = torch.linspace(0, 2 * np.pi, self.dimension, device=self.device)
        field *= torch.exp(-r / self.mass)  # Gravitational falloff
        return field

    def update_position(self, dt, force):
        """Update position using Newtonian physics"""
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class PhysicalTensorUniverse:
    def __init__(self, size=50, num_singularities=100, dimension=128, device='cpu'):
        self.G = 6.67430e-11  # Gravitational constant
        self.size = size
        self.dimension = dimension
        self.device = device
        self.space = torch.zeros((size, size, size), device=self.device)
        self.singularities = []
        self.initialize_singularities(num_singularities)

    def initialize_singularities(self, num):
        """Initialize singularities with random positions and masses"""
        for _ in range(num):
            position = torch.tensor(np.random.rand(3) * self.size, dtype=torch.float32, device=self.device)
            mass = torch.distributions.Exponential(1.0).sample().item()  # Random masses
            self.singularities.append(
                PhysicalTensorSingularity(
                    dimension=self.dimension,
                    position=position,
                    mass=mass,
                    device=self.device
                )
            )

    def calculate_gravity(self, pos1, pos2, m1, m2):
        """Calculate gravitational force between two points"""
        r = pos2 - pos1
        distance = torch.norm(r) + 1e-10
        force_magnitude = self.G * m1 * m2 / (distance ** 2)
        return force_magnitude * r / distance

    def update_tensor_interactions(self):
        """Update tensor field interactions using vectorized operations"""
        positions = torch.stack([s.position for s in self.singularities])  # Shape: [N, 3]
        masses = torch.tensor([s.mass for s in self.singularities], device=self.device)  # Shape: [N]

        # Calculate pairwise distances and forces
        delta = positions.unsqueeze(1) - positions.unsqueeze(0)  # Shape: [N, N, 3]
        distance = torch.norm(delta, dim=2) + 1e-10  # Shape: [N, N]
        force_magnitude = self.G * masses.unsqueeze(1) * masses.unsqueeze(0) / (distance ** 2)  # Shape: [N, N]
        force_direction = delta / distance.unsqueeze(2)  # Shape: [N, N, 3]
        force = torch.sum(force_magnitude.unsqueeze(2) * force_direction, dim=1)  # Shape: [N, 3]

        # Apply tensor field interactions
        fields = torch.stack([s.field for s in self.singularities])  # Shape: [N, D]
        field_interaction = torch.tanh(torch.matmul(fields, fields.T))  # Shape: [N, N]
        force *= (1 + torch.mean(field_interaction, dim=1)).unsqueeze(1)

        # Update positions
        for i, singularity in enumerate(self.singularities):
            singularity.update_position(dt=0.1, force=force[i])

    def update_space(self):
        """Update 3D space based on singularity positions and fields"""
        self.space.fill_(0)
        x = torch.linspace(0, self.size, self.size, device=self.device)
        y = torch.linspace(0, self.size, self.size, device=self.device)
        z = torch.linspace(0, self.size, self.size, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        for s in self.singularities:
            # Calculate distance from singularity to each point
            R = torch.sqrt((X - s.position[0]) ** 2 +
                          (Y - s.position[1]) ** 2 +
                          (Z - s.position[2]) ** 2)
            # Add field influence
            self.space += s.mass / (R + 1) * torch.mean(s.field)

    def detect_structures(self):
        """Detect galaxy-like structures using clustering"""
        structures = []
        density_threshold = torch.mean(self.space) + torch.std(self.space)

        # Find high density regions
        dense_regions = self.space > density_threshold

        # Convert to NumPy for clustering
        dense_indices = torch.nonzero(dense_regions, as_tuple=False).cpu().numpy()

        if dense_indices.size == 0:
            return structures

        # Basic clustering using scipy
        labeled_array, num_features = label(dense_regions.cpu().numpy())

        for i in range(1, num_features + 1):
            region = torch.nonzero(labeled_array == i, as_tuple=False).numpy()
            if region.size == 0:
                continue
            center = np.mean(region, axis=0)
            mass = torch.sum(self.space[labeled_array == i]).item()
            size = region.shape[0]
            structures.append({
                'center': center,
                'mass': mass,
                'size': size
            })

        return structures

def create_density_slice_image(universe, slice_axis='z'):
    """
    Create a 2D density slice image from the 3D space.

    Args:
        universe (PhysicalTensorUniverse): The simulation universe.
        slice_axis (str): The axis to slice ('x', 'y', or 'z').

    Returns:
        PIL.Image: The rendered density slice image.
    """
    # Select the slice axis and compute the middle slice
    size = universe.size
    if slice_axis == 'x':
        slice_index = size // 2
        density_slice = universe.space[slice_index, :, :].cpu().numpy()
    elif slice_axis == 'y':
        slice_index = size // 2
        density_slice = universe.space[:, slice_index, :].cpu().numpy()
    else:  # 'z'
        slice_index = size // 2
        density_slice = universe.space[:, :, slice_index].cpu().numpy()

    # Normalize the density slice for visualization
    density_normalized = (density_slice - density_slice.min()) / (density_slice.max() - density_slice.min())
    density_normalized = np.uint8(255 * density_normalized)

    # Create a PIL image
    image = Image.fromarray(density_normalized)
    image = image.resize((1024, 1024))  # Stretch to 1024x1024

    return image

def run_simulation(universe_size, num_singularities, simulation_steps):
    """
    Run the simulation and capture density slice images.

    Args:
        universe_size (int): Size of the universe.
        num_singularities (int): Number of singularities.
        simulation_steps (int): Number of simulation steps.

    Returns:
        str: The file path to the generated GIF.
    """
    # Determine the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the universe
    universe = PhysicalTensorUniverse(
        size=universe_size,
        num_singularities=num_singularities,
        device=device
    )

    images = []
    interval = max(simulation_steps // 100, 1)  # Capture up to 100 frames

    for step in range(simulation_steps):
        universe.update_tensor_interactions()
        universe.update_space()

        if step % interval == 0:
            img = create_density_slice_image(universe, slice_axis='z')
            images.append(img)

    # Save images as a GIF
    if not images:
        # Ensure at least one image is captured
        img = create_density_slice_image(universe, slice_axis='z')
        images.append(img)

    # Use a temporary file to save the GIF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as temp_file:
        images[0].save(
            temp_file,
            format='GIF',
            append_images=images[1:],
            save_all=True,
            duration=100,  # Duration between frames in ms
            loop=0
        )
        gif_path = temp_file.name

    return gif_path

def create_gradio_interface():
    """
    Create and launch the Gradio interface.

    Returns:
        Gradio.Interface: The Gradio interface instance.
    """
    interface = gr.Interface(
        fn=run_simulation,
        inputs=[
            gr.Slider(10, 200, step=10, label="Universe Size", value=50),
            gr.Slider(10, 500, step=10, label="Number of Singularities", value=100),
            gr.Slider(10, 1000, step=10, label="Simulation Steps", value=100)
        ],
        outputs=gr.File(label="Density Slice Evolution"),
        title="Fast Tensor Universe Simulation",
        description="""
        Simulate a universe with tensor singularities.
        - **Universe Size:** Determines the spatial dimensions.
        - **Number of Singularities:** Entities affecting the density field.
        - **Simulation Steps:** Number of iterations to run the simulation.

        The density slice is visualized as an evolving GIF.
        """
    )
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
