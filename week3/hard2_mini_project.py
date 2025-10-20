"""
End-to-End Mini Project: Image Processing Pipeline using NumPy Fundamentals
A complete image processing and analysis system
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import urllib.request
import os
from PIL import Image
import time

class ImageProcessingPipeline:
    """
    A comprehensive image processing pipeline using NumPy fundamentals
    """
    
    def __init__(self):
        self.images = {}
        self.processed_images = {}
        self.analysis_results = {}
    
    def load_sample_images(self):
        """
        Load sample images for processing
        """
        print("Loading sample images...")
        try:
            # Load sample images from sklearn
            dataset = load_sample_images()
            self.images['original'] = [img for img in dataset.images]
            print(f"Loaded {len(self.images['original'])} sample images")
            
        except Exception as e:
            print(f"Error loading sample images: {e}")
            print("Creating synthetic images...")
            self._create_synthetic_images()
    
    def _create_synthetic_images(self):
        """
        Create synthetic images for demonstration
        """
        # Create various synthetic images
        images = []
        
        # Gradient image
        gradient = np.linspace(0, 255, 256).astype(np.uint8)
        gradient_img = np.tile(gradient, (256, 1))
        images.append(np.stack([gradient_img] * 3, axis=-1))  # RGB
        
        # Checkerboard pattern
        checkerboard = np.zeros((256, 256, 3), dtype=np.uint8)
        checkerboard[::32, ::32] = 255
        images.append(checkerboard)
        
        # Random noise image
        noise_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        images.append(noise_img)
        
        # Circular pattern
        y, x = np.ogrid[-128:128, -128:128]
        mask = x*x + y*y <= 128*128
        circle_img = np.zeros((256, 256, 3), dtype=np.uint8)
        circle_img[mask] = [255, 0, 0]  # Red circle
        images.append(circle_img)
        
        self.images['original'] = images
        print(f"Created {len(images)} synthetic images")
    
    def preprocess_images(self):
        """
        Preprocess images using NumPy operations
        """
        print("\n=== IMAGE PREPROCESSING ===")
        
        original_images = self.images['original']
        processed = []
        
        for i, img in enumerate(original_images):
            print(f"Processing image {i+1}...")
            
            # Convert to float for processing
            img_float = img.astype(np.float32) / 255.0
            
            # 1. Normalize image
            normalized = self._normalize_image(img_float)
            
            # 2. Convert to grayscale
            grayscale = self._rgb_to_grayscale(normalized)
            
            # 3. Apply Gaussian blur
            blurred = self._gaussian_blur(grayscale)
            
            # 4. Edge detection
            edges = self._sobel_edge_detection(blurred)
            
            processed.append({
                'original': img,
                'normalized': normalized,
                'grayscale': grayscale,
                'blurred': blurred,
                'edges': edges
            })
        
        self.processed_images = processed
        print("Image preprocessing completed!")
    
    def _normalize_image(self, image):
        """
        Normalize image using NumPy
        """
        # Normalize each channel separately
        normalized = np.zeros_like(image)
        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel]
            normalized[:, :, channel] = (channel_data - np.min(channel_data)) / (
                np.max(channel_data) - np.min(channel_data) + 1e-8)
        return normalized
    
    def _rgb_to_grayscale(self, image):
        """
        Convert RGB image to grayscale using luminance formula
        """
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    
    def _gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian blur using NumPy
        """
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution
        padded = np.pad(image, kernel_size//2, mode='reflect')
        blurred = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                blurred[i, j] = np.sum(region * kernel)
        
        return blurred
    
    def _create_gaussian_kernel(self, size, sigma):
        """
        Create Gaussian kernel
        """
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * 
                         np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
    
    def _sobel_edge_detection(self, image):
        """
        Apply Sobel edge detection
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution
        padded = np.pad(image, 1, mode='reflect')
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+3, j:j+3]
                grad_x[i, j] = np.sum(region * sobel_x)
                grad_y[i, j] = np.sum(region * sobel_y)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude / np.max(gradient_magnitude)
    
    def analyze_images(self):
        """
        Perform comprehensive image analysis
        """
        print("\n=== IMAGE ANALYSIS ===")
        
        analysis_results = []
        
        for i, processed in enumerate(self.processed_images):
            print(f"Analyzing image {i+1}...")
            
            original = processed['original']
            grayscale = processed['grayscale']
            edges = processed['edges']
            
            # 1. Basic statistics
            stats = self._compute_image_statistics(original, grayscale)
            
            # 2. Histogram analysis
            histograms = self._compute_histograms(original)
            
            # 3. Texture analysis
            texture_features = self._analyze_texture(grayscale)
            
            # 4. Edge analysis
            edge_features = self._analyze_edges(edges)
            
            # 5. Color analysis
            color_features = self._analyze_colors(original)
            
            analysis_results.append({
                'stats': stats,
                'histograms': histograms,
                'texture': texture_features,
                'edges': edge_features,
                'colors': color_features
            })
        
        self.analysis_results = analysis_results
        print("Image analysis completed!")
    
    def _compute_image_statistics(self, original, grayscale):
        """
        Compute basic image statistics
        """
        stats = {
            'shape': original.shape,
            'dtype': original.dtype,
            'min_value': np.min(original),
            'max_value': np.max(original),
            'mean_value': np.mean(original),
            'std_value': np.std(original),
            'grayscale_mean': np.mean(grayscale),
            'grayscale_std': np.std(grayscale)
        }
        return stats
    
    def _compute_histograms(self, image):
        """
        Compute RGB histograms
        """
        histograms = {}
        for channel, color in enumerate(['red', 'green', 'blue']):
            histograms[color] = np.histogram(image[:, :, channel], bins=256, range=(0, 255))[0]
        return histograms
    
    def _analyze_texture(self, image):
        """
        Analyze image texture using statistical measures
        """
        # Compute local binary pattern-like features
        texture_features = {
            'contrast': np.std(image),
            'smoothness': 1 - 1/(1 + np.var(image)),
            'uniformity': np.sum(np.histogram(image, bins=32, density=True)[0]**2),
            'entropy': -np.sum(np.histogram(image, bins=32, density=True)[0] * 
                              np.log2(np.histogram(image, bins=32, density=True)[0] + 1e-8))
        }
        return texture_features
    
    def _analyze_edges(self, edge_image):
        """
        Analyze edge characteristics
        """
        edge_threshold = 0.1
        strong_edges = edge_image > edge_threshold
        
        edge_features = {
            'edge_density': np.mean(strong_edges),
            'total_edges': np.sum(strong_edges),
            'mean_edge_strength': np.mean(edge_image[strong_edges]) if np.any(strong_edges) else 0,
            'edge_std': np.std(edge_image[strong_edges]) if np.any(strong_edges) else 0
        }
        return edge_features
    
    def _analyze_colors(self, image):
        """
        Analyze color characteristics
        """
        # Convert to HSV color space
        image_float = image.astype(np.float32) / 255.0
        hsv_image = self._rgb_to_hsv(image_float)
        
        color_features = {
            'dominant_hue': np.mean(hsv_image[:, :, 0]),
            'color_saturation': np.mean(hsv_image[:, :, 1]),
            'color_brightness': np.mean(hsv_image[:, :, 2]),
            'color_std': np.std(image_float, axis=(0, 1))
        }
        return color_features
    
    def _rgb_to_hsv(self, rgb):
        """
        Convert RGB to HSV color space
        """
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        diff = maxc - minc
        
        h = np.zeros_like(maxc)
        s = np.zeros_like(maxc)
        v = maxc
        
        # Hue calculation
        mask = diff != 0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        # Red is max
        idx = (maxc == r) & mask
        h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
        
        # Green is max
        idx = (maxc == g) & mask
        h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
        
        # Blue is max
        idx = (maxc == b) & mask
        h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360
        
        # Saturation calculation
        s[mask] = diff[mask] / maxc[mask]
        
        return np.stack([h/360.0, s, v], axis=-1)
    
    def apply_image_compression(self):
        """
        Apply image compression using PCA
        """
        print("\n=== IMAGE COMPRESSION USING PCA ===")
        
        compressed_images = []
        
        for i, processed in enumerate(self.processed_images):
            print(f"Compressing image {i+1}...")
            
            original = processed['original']
            grayscale = processed['grayscale']
            
            # Reshape image for PCA
            if len(original.shape) == 3:  # Color image
                flattened = original.reshape(-1, original.shape[-1])
            else:  # Grayscale
                flattened = grayscale.reshape(-1, 1)
            
            # Apply PCA for compression
            n_components = min(50, flattened.shape[1], flattened.shape[0])
            pca = PCA(n_components=n_components)
            compressed = pca.fit_transform(flattened)
            reconstructed = pca.inverse_transform(compressed)
            
            # Reshape back to image
            if len(original.shape) == 3:
                reconstructed_img = reconstructed.reshape(original.shape)
            else:
                reconstructed_img = reconstructed.reshape(grayscale.shape)
            
            # Calculate compression ratio
            original_size = original.nbytes
            compressed_size = compressed.nbytes + pca.components_.nbytes + pca.mean_.nbytes
            compression_ratio = original_size / compressed_size
            
            compressed_images.append({
                'compressed': compressed,
                'reconstructed': reconstructed_img,
                'pca': pca,
                'compression_ratio': compression_ratio,
                'explained_variance': np.sum(pca.explained_variance_ratio_)
            })
        
        self.compressed_images = compressed_images
        print("Image compression completed!")
    
    def image_segmentation(self):
        """
        Perform image segmentation using K-means clustering
        """
        print("\n=== IMAGE SEGMENTATION ===")
        
        segmented_images = []
        
        for i, processed in enumerate(self.processed_images):
            print(f"Segmenting image {i+1}...")
            
            original = processed['original']
            
            # Reshape image for clustering
            pixels = original.reshape(-1, 3)
            
            # Apply K-means clustering
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Create segmented image
            segmented = kmeans.cluster_centers_[labels].reshape(original.shape).astype(np.uint8)
            
            segmented_images.append({
                'segmented': segmented,
                'labels': labels.reshape(original.shape[:2]),
                'kmeans': kmeans,
                'n_clusters': n_clusters
            })
        
        self.segmented_images = segmented_images
        print("Image segmentation completed!")
    
    def visualize_results(self):
        """
        Create comprehensive visualization of all results
        """
        print("\n=== CREATING VISUALIZATIONS ===")
        
        n_images = len(self.processed_images)
        fig, axes = plt.subplots(n_images, 6, figsize=(20, 4*n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            processed = self.processed_images[i]
            analysis = self.analysis_results[i]
            
            # Original image
            axes[i, 0].imshow(processed['original'])
            axes[i, 0].set_title(f'Image {i+1}\nOriginal')
            axes[i, 0].axis('off')
            
            # Grayscale
            axes[i, 1].imshow(processed['grayscale'], cmap='gray')
            axes[i, 1].set_title('Grayscale')
            axes[i, 1].axis('off')
            
            # Edges
            axes[i, 2].imshow(processed['edges'], cmap='hot')
            axes[i, 2].set_title('Edge Detection')
            axes[i, 2].axis('off')
            
            # Histograms
            colors = ['red', 'green', 'blue']
            for j, color in enumerate(colors):
                axes[i, 3].plot(analysis['histograms'][color], color=color, alpha=0.7)
            axes[i, 3].set_title('RGB Histograms')
            axes[i, 3].set_xlabel('Pixel Value')
            axes[i, 3].set_ylabel('Frequency')
            
            # Compressed image (if available)
            if hasattr(self, 'compressed_images'):
                axes[i, 4].imshow(self.compressed_images[i]['reconstructed'].astype(np.uint8))
                axes[i, 4].set_title(f"Compressed\nRatio: {self.compressed_images[i]['compression_ratio']:.2f}x")
                axes[i, 4].axis('off')
            
            # Segmented image (if available)
            if hasattr(self, 'segmented_images'):
                axes[i, 5].imshow(self.segmented_images[i]['segmented'])
                axes[i, 5].set_title(f'Segmented\n{self.segmented_images[i]["n_clusters"]} clusters')
                axes[i, 5].axis('off')
        
        plt.tight_layout()
        plt.savefig('image_processing_pipeline.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create analysis summary visualization
        self._create_analysis_summary()
    
    def _create_analysis_summary(self):
        """
        Create summary visualization of analysis results
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Texture features comparison
        texture_features = ['contrast', 'smoothness', 'uniformity', 'entropy']
        n_images = len(self.analysis_results)
        
        for j, feature in enumerate(texture_features):
            values = [analysis['texture'][feature] for analysis in self.analysis_results]
            axes[0, 0].bar(range(n_images), values, alpha=0.7, label=feature)
        axes[0, 0].set_title('Texture Features')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Feature Value')
        axes[0, 0].legend()
        
        # Edge features comparison
        edge_features = ['edge_density', 'total_edges', 'mean_edge_strength']
        for j, feature in enumerate(edge_features):
            values = [analysis['edges'][feature] for analysis in self.analysis_results]
            axes[0, 1].bar(range(n_images), values, alpha=0.7, label=feature)
        axes[0, 1].set_title('Edge Features')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].legend()
        
        # Color features
        color_features = ['dominant_hue', 'color_saturation', 'color_brightness']
        for j, feature in enumerate(color_features):
            values = [analysis['colors'][feature] for analysis in self.analysis_results]
            axes[0, 2].bar(range(n_images), values, alpha=0.7, label=feature)
        axes[0, 2].set_title('Color Features')
        axes[0, 2].set_xlabel('Image Index')
        axes[0, 2].set_ylabel('Feature Value')
        axes[0, 2].legend()
        
        # Compression results (if available)
        if hasattr(self, 'compressed_images'):
            compression_ratios = [comp['compression_ratio'] for comp in self.compressed_images]
            explained_variance = [comp['explained_variance'] for comp in self.compressed_images]
            
            axes[1, 0].bar(range(n_images), compression_ratios, color='green', alpha=0.7)
            axes[1, 0].set_title('Compression Ratios')
            axes[1, 0].set_xlabel('Image Index')
            axes[1, 0].set_ylabel('Compression Ratio')
            
            axes[1, 1].bar(range(n_images), explained_variance, color='orange', alpha=0.7)
            axes[1, 1].set_title('Explained Variance')
            axes[1, 1].set_xlabel('Image Index')
            axes[1, 1].set_ylabel('Variance Ratio')
        
        # Performance metrics
        stats_keys = ['mean_value', 'std_value', 'grayscale_mean', 'grayscale_std']
        stats_data = np.array([[analysis['stats'][key] for key in stats_keys] 
                              for analysis in self.analysis_results])
        
        im = axes[1, 2].imshow(stats_data.T, cmap='viridis', aspect='auto')
        axes[1, 2].set_title('Statistical Features Heatmap')
        axes[1, 2].set_xlabel('Image Index')
        axes[1, 2].set_ylabel('Feature Type')
        axes[1, 2].set_yticks(range(len(stats_keys)))
        axes[1, 2].set_yticklabels(stats_keys)
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_pipeline(self):
        """
        Run the complete image processing pipeline
        """
        print("=== IMAGE PROCESSING PIPELINE STARTED ===\n")
        start_time = time.time()
        
        # Step 1: Load images
        self.load_sample_images()
        
        # Step 2: Preprocess images
        self.preprocess_images()
        
        # Step 3: Analyze images
        self.analyze_images()
        
        # Step 4: Apply compression
        self.apply_image_compression()
        
        # Step 5: Perform segmentation
        self.image_segmentation()
        
        # Step 6: Visualize results
        self.visualize_results()
        
        end_time = time.time()
        print(f"\n=== PIPELINE COMPLETED IN {end_time - start_time:.2f} SECONDS ===")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """
        Print pipeline summary
        """
        print("\n=== PIPELINE SUMMARY ===")
        print(f"Processed {len(self.processed_images)} images")
        print(f"Applied {len(self.processed_images[0]) - 1} preprocessing steps per image")
        print(f"Computed {len(self.analysis_results[0])} analysis categories per image")
        
        if hasattr(self, 'compressed_images'):
            avg_compression = np.mean([comp['compression_ratio'] for comp in self.compressed_images])
            avg_variance = np.mean([comp['explained_variance'] for comp in self.compressed_images])
            print(f"Average compression ratio: {avg_compression:.2f}x")
            print(f"Average explained variance: {avg_variance:.3f}")
        
        print("\nNumPy Fundamentals Used:")
        print("✓ Array creation and manipulation")
        print("✓ Vectorized operations")
        print("✓ Broadcasting")
        print("✓ Universal functions (ufuncs)")
        print("✓ Linear algebra (PCA)")
        print("✓ Statistical operations")
        print("✓ Memory-efficient operations")
        print("✓ Image processing algorithms")

def main():
    """
    Main function to run the image processing pipeline
    """
    pipeline = ImageProcessingPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()