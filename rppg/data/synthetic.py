import numpy as np
#from utils import to_categorical

# Adapted from 
# 3D Convolutional Neural Networks for Remote Pulse Rate Measurement
#  and Mapping from Facial Video by Bousefsaf et al.

class DataGen():
    def __init__(self, heart_rate_low=55, heart_rate_high=240,
                 heart_rate_resolution=2.5, length_vid=60, img_width=25,
                 img_height=25, img_channels=1, sampling=1/30,
                 no_videos_by_class=300, seed=None):
        # constants
        self.heart_rates = np.arange(
            heart_rate_low, heart_rate_high, heart_rate_resolution)
        self.length_vid = length_vid
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.sampling = sampling
        self.no_videos_by_class = no_videos_by_class
        self.rng = np.random.default_rng(seed)

        self.no_classes = len(self.heart_rates)
        self.t = np.linspace(0, self.length_vid *
                             self.sampling - self.sampling, self.length_vid)
        self.freqs = self.heart_rates / 60

        # prepare labels and label categories
        
        # for categorical cross entropy doesnt need one hot vectors
        # pytorch expects long dtypes
        #labels = np.arange(0, self.no_classes + 1, 1, dtype='uint8')
        #self.labels_cat = to_categorical(labels, self.no_classes + 1)
        
        labels = np.arange(0, self.no_classes + 1, 1, dtype=np.int64)
        self.labels_cat = labels
        
        # coefficients for the fitted-ppg method
        self.coeff = {
        'a0': 0.440240602542388,
        'a1': -0.334501803331783,
        'b1': -0.198990393984879,
        'a2': -0.050159136439220,
        'b2': 0.099347477830878,
        'w': 2 * np.pi     
        }

        # Tendencies (linear, 2nd order, 3rd order)
        self.tendencies_min = (-3, -1, -1)
        self.tendencies_max = (3, 1, 1)
        self.tendencies_order = (1, 2, 3)

    def gen_trend(self, length, order, min, max, offset):
        if (order == 1):   # linear
            tend = np.linspace(min, max, length)

        elif (order == 2):  # quadratic
            if (offset == 0):
                tend = np.linspace(0, 1, length)
                tend = tend*tend
                tend = tend-min
                tend = max*tend/np.max(tend)

            else:
                tend = tend = np.linspace(-0.5, 0.5, length)
                tend = tend*tend
                tend = tend-min
                tend = 0.5*max*tend/np.max(tend)

        elif (order == 3):  # cubic
            if (offset == 0):
                tend = np.linspace(0, 1, length)
                tend = tend*tend*tend
                tend = tend-min
                tend = max*tend/np.max(tend)

            else:
                tend = tend = np.linspace(-0.5, 0.5, length)
                tend = tend*tend*tend
                tend = tend-min
                tend = 0.5*max*tend/np.max(tend)
        return tend

    def random_trend(self):
        index = self.rng.integers(len(self.tendencies_order))
        return self.gen_trend(
            self.length_vid,
            self.tendencies_order[index],
            0,
            self.rng.uniform(self.tendencies_min[index], self.tendencies_max[index]),
            self.rng.integers(2),
        )

    def gen_signal(self):
        # The model expects N, C, H, W, T, with the temporal axis last.
        count = (self.no_classes + 1) * self.no_videos_by_class
        video_shape = (self.img_height, self.img_width, self.length_vid)
        x = np.empty((count, self.img_channels, *video_shape), dtype=np.float32)
        y = np.empty(count, dtype=np.int64)
        area = self.img_height * self.img_width
        coeff = self.coeff

        index = 0
        for _ in range(self.no_videos_by_class):
            for label, freq in enumerate(self.freqs):
                phase = self.rng.integers(0, 33) * self.sampling
                angle = (self.t + phase) * coeff["w"] * freq
                signal = (
                    coeff["a0"] + coeff["a1"] * np.cos(angle) + coeff["b1"] * np.sin(angle)
                    + coeff["a2"] * np.cos(2 * angle) + coeff["b2"] * np.sin(2 * angle)
                )
                signal -= signal.min()
                signal /= signal.max()
                signal += self.random_trend()
                signal -= signal.min()

                amplitude = self.rng.uniform(1.5, 4)
                noise_energy = amplitude * 0.25 * self.rng.uniform(1, 10) / 100
                noise = self.rng.normal(0.5, 0.25, video_shape)
                pixels = 255 * (amplitude * (signal / area) + noise * noise_energy)
                x[index] = np.maximum(pixels, 0).astype(np.uint8) / 255.0
                x[index] -= x[index].mean()
                y[index] = self.labels_cat[label]
                index += 1

        for _ in range(self.no_videos_by_class):
            x[index] = self.rng.normal(size=video_shape) / 50 + self.random_trend() / area
            x[index] -= x[index].mean()
            y[index] = self.labels_cat[self.no_classes]
            index += 1
        return x, y
