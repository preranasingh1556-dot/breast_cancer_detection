"""
gradcam.py
Grad-CAM implementation for explainability on a ResNet-based classifier.

Usage:
    from gradcam import GradCAM

    cam_engine = GradCAM(model, target_layer=model.layer4)
    heatmap, pred_class = cam_engine.generate(input_tensor)
"""

import matplotlib
import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    Grad-CAM for CNNs (built for torchvision ResNet models, e.g. model.layer4).

    Works by:
      1. Hooking the target conv layer to capture its forward activations
         and backward gradients.
      2. Running a forward pass, then backpropagating from the predicted
         (or specified) class score.
      3. Weighting each activation channel by the average gradient flowing
         into it (global-average-pooled gradients), then summing + ReLU-ing
         to produce a coarse localization heatmap.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Default to the last conv block of a torchvision ResNet
        if target_layer is None:
            if hasattr(model, "layer4"):
                target_layer = model.layer4
            else:
                raise ValueError(
                    "Could not auto-detect target_layer. "
                    "Pass it explicitly, e.g. GradCAM(model, model.layer4)."
                )

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Args:
            input_tensor: preprocessed image tensor, shape (1, C, H, W)
            class_idx: which class to explain. If None, uses the model's
                       predicted (argmax) class.

        Returns:
            heatmap: numpy array, shape (H, W), values in [0, 1]
            class_idx: the class index that was explained
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # (1, num_classes)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        
        score = output[0, class_idx]
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        score.backward()

        # Global-average-pool the gradients over spatial dims -> per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation channels
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to input resolution
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx,probs

    def overlay_on_image(self, heatmap, original_image, alpha=0.4, colormap="jet"):
        """
        Overlays the Grad-CAM heatmap on top of the original image.

        Args:
            heatmap: numpy array (H, W) in [0, 1], from generate()
            original_image: numpy array (H, W, 3) in [0, 255] or [0, 1]
            alpha: blend strength of the heatmap
            colormap: matplotlib colormap name

        Returns:
            numpy array (H, W, 3), uint8, ready for display
        """
        if original_image.dtype != np.uint8:
            original_image = np.uint8(255 * original_image / original_image.max())

        cmap = matplotlib.colormaps[colormap]
        colored_heatmap = cmap(heatmap)[:, :, :3]  # drop alpha channel
        colored_heatmap = np.uint8(255 * colored_heatmap)

        overlay = np.uint8(
            colored_heatmap * alpha + original_image * (1 - alpha)
        )
        return overlay
