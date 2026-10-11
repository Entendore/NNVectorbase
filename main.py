import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Digits": load_digits()
}

# ---------------- Plot Widget with pulsing ----------------
class PlotWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = []
        self.performances = []
        self.highlight_idx = None
        self.neighbors = []
        self.pulse_phase = 0.0
        self.pulse_speed = 0.15
        Clock.schedule_interval(self.update_pulse, 1/60)

    def draw_points(self, points, performances, highlight_idx=None, neighbors=None):
        self.points = points
        self.performances = performances
        self.highlight_idx = highlight_idx

        # Ensure neighbors is a Python list
        if neighbors is None:
            self.neighbors = []
        elif isinstance(neighbors, np.ndarray):
            self.neighbors = neighbors.tolist()
        else:
            self.neighbors = neighbors

        self._draw()

    def _draw(self):
        self.canvas.clear()
        if self.points.shape[0] == 0:
            return

        with self.canvas:
            xs = self.points[:,0]
            ys = self.points[:,1]
            xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-8) * self.width
            ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8) * self.height

            # Draw all points
            for i, (x, y) in enumerate(zip(xs, ys)):
                perf = self.performances[i]
                Color(0, perf, 1 - perf)
                Ellipse(pos=(x-5, y-5), size=(10, 10))

            # Pulse factor and color
            pulse_factor = 1.0 + 0.3 * np.sin(self.pulse_phase)
            color_pulse = 0.5 + 0.5 * np.sin(self.pulse_phase)

            # Draw neighbors
            if self.neighbors and self.highlight_idx is not None:
                for idx in self.neighbors:
                    Color(1, 0.5 * color_pulse, 0)
                    size = 12 * pulse_factor
                    Ellipse(pos=(xs[idx]-size/2, ys[idx]-size/2), size=(size, size))
                    Line(points=[xs[self.highlight_idx], ys[self.highlight_idx], xs[idx], ys[idx]], width=1.2)

            # Draw query
            if self.highlight_idx is not None:
                Color(1, color_pulse, color_pulse)
                size = 16 * pulse_factor
                Ellipse(pos=(xs[self.highlight_idx]-size/2, ys[self.highlight_idx]-size/2), size=(size, size))

    def update_pulse(self, dt):
        # Only update pulse if query and neighbors exist
        if self.highlight_idx is not None and self.neighbors and len(self.neighbors) > 0:
            self.pulse_phase += self.pulse_speed
            self._draw()

# ---------------- Main App ----------------
class AnimatedVectorDBApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical')

        # Dataset selector
        self.spinner = Spinner(text='Select Dataset', values=list(datasets.keys()))
        self.root.add_widget(self.spinner)

        # Buttons
        self.generate_button = Button(text="Generate Random Networks")
        self.generate_button.bind(on_press=self.generate_random_networks)
        self.root.add_widget(self.generate_button)

        self.animate_button = Button(text="Start Animation")
        self.animate_button.bind(on_press=self.start_animation)
        self.root.add_widget(self.animate_button)

        self.stop_button = Button(text="Stop Animation")
        self.stop_button.bind(on_press=self.stop_animation)
        self.root.add_widget(self.stop_button)

        # Neighbor count selector
        self.neighbor_spinner = Spinner(text='5', values=[str(i) for i in range(1, 21)],
                                        size_hint=(None, None), size=(100, 44))
        self.neighbor_spinner.bind(text=self.update_neighbors_count)
        self.root.add_widget(self.neighbor_spinner)
        self.num_neighbors = 5

        # Status label
        self.status_label = Label(text="Generate networks first.")
        self.root.add_widget(self.status_label)

        # Plot widget
        self.plot_widget = PlotWidget()
        self.root.add_widget(self.plot_widget)

        # Storage
        self.dataset_embeddings = {}
        self.dataset_embeddings_2d = {}
        self.dataset_performances = {}
        self.animation_index = 0
        self.animation_datasets = list(datasets.keys())
        self.animation_timer = None
        self.current_query_idx = None
        self.current_neighbors = None

        return self.root

    # ---------------- Generate Random Networks ----------------
    def generate_random_networks(self, instance):
        self.status_label.text = "Generating networks..."
        for dataset_name, data in datasets.items():
            X = data.data.astype('float32')
            y = data.target
            n_classes = len(np.unique(y))

            embeddings = []
            performances = []

            for i in range(100):
                nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1, random_state=i, warm_start=False)
                try:
                    nn._validate_input(X, y)
                    nn._random_state = np.random.RandomState(i)
                    nn._initialize(X.shape[1], n_classes)
                    out = nn._forward_pass(X)
                    embedding = np.mean(out, axis=0).astype('float32')
                    accuracy = np.random.rand()
                except Exception:
                    out = np.random.rand(X.shape[0], n_classes)
                    embedding = out.mean(axis=0).astype('float32')
                    accuracy = np.random.rand()

                embeddings.append(embedding)
                performances.append(accuracy)

            self.dataset_embeddings[dataset_name] = np.array(embeddings)
            self.dataset_performances[dataset_name] = performances
            if embeddings[0].shape[0] > 2:
                self.dataset_embeddings_2d[dataset_name] = PCA(n_components=2).fit_transform(np.array(embeddings))
            else:
                self.dataset_embeddings_2d[dataset_name] = np.array(embeddings)

        self.status_label.text = "Generated 100 random networks for all datasets."
        self.plot_current_dataset()

    # ---------------- Plot Current Dataset ----------------
    def plot_current_dataset(self, highlight_idx=None, neighbors=None):
        dataset_name = self.animation_datasets[self.animation_index]
        vectors_2d = self.dataset_embeddings_2d[dataset_name]
        performances = self.dataset_performances[dataset_name]
        self.plot_widget.draw_points(vectors_2d, performances, highlight_idx, neighbors)

    # ---------------- Animation ----------------
    def start_animation(self, instance):
        if not self.dataset_embeddings:
            self.status_label.text = "Please generate networks first!"
            return
        if self.animation_timer:
            self.animation_timer.cancel()
        self.animation_index = 0
        self.animation_timer = Clock.schedule_interval(self.animate_step, 2.0)

    def stop_animation(self, instance):
        if self.animation_timer:
            self.animation_timer.cancel()
            self.animation_timer = None

    def animate_step(self, dt):
        dataset_name = self.animation_datasets[self.animation_index]
        vectors = self.dataset_embeddings[dataset_name]
        query_idx = np.random.randint(len(vectors))
        distances = np.linalg.norm(vectors - vectors[query_idx], axis=1)
        nearest_indices = np.argsort(distances)[1:self.num_neighbors+1]

        self.current_query_idx = query_idx
        self.current_neighbors = nearest_indices.tolist()

        self.status_label.text = (
            f"Dataset: {dataset_name}\n"
            f"Query index: {query_idx}\n"
            f"Nearest neighbors: {nearest_indices}"
        )

        vectors_2d = self.dataset_embeddings_2d[dataset_name]
        performances = self.dataset_performances[dataset_name]
        self.plot_widget.draw_points(vectors_2d, performances, query_idx, nearest_indices)

        self.animation_index = (self.animation_index + 1) % len(self.animation_datasets)

    def update_neighbors_count(self, spinner, text):
        self.num_neighbors = int(text)

# ---------------- Run App ----------------
if __name__ == "__main__":
    AnimatedVectorDBApp().run()
