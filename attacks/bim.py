import torch
import torch.nn as functional
import numpy as np
from typing import Tuple, List, Optional


class BIM:
    def __init__(self, epsilon: float,  input_label: int, model: functional, steps: int):
        self.model = model
        self.epsilon = epsilon
        self.target = input_label
        self.steps = steps
        self.clip_value = 1.0

    def basic_iterative(self, image: torch.Tensor, targeted = False) -> torch.Tensor:
        target_tensor = torch.zeros((1000,1))[self.target,0] = 1
        pert = torch.zeros_like(image)

        image.requires_grad = True

        for it in range(self.steps):

            adv_image = image + pert
            pred = self.model(adv_image)

            loss = functional.CrossEntropyLoss(torch.max(pred).unsqueeze(0), target_tensor)
            loss.backward()

            if it%1 == 0:
                print(f"iteration: {it}, loss: {loss.data.numpy()}")
            
            grads_sign = -1*torch.sign(image.grad.data) if targeted else torch.sign(image.grad.data)

            grads_sign.requires_grad = False
            image.requires_grad = False

            pert.add(grads_sign * self.epsilon)
            pert = torch.clamp(pert,-self.clip_value, self.clip_value)
        
        return pert

    
    def generate(self, images: List[torch.Tensor], targeted = False) -> List[np.array]:

        perturbations = np.zeros_like(images)

        for ind,img in enumerate(images):
            perturbations[ind] = self.basic_iterative(img, targeted).numpy()
        
        return perturbations