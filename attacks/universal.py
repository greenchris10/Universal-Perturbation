import torch
import torch.nn as functional
import numpy as np
from typing import Tuple, List, Optional

class Universal:

    def __init__(self, epsilon: float,  input_label: int, model: functional, steps: int):
        self.model = model
        self.epsilon = epsilon
        self.target = input_label
        self.steps = steps
        self.clip_value = 1.0
    
    def generate(self, image: List[torch.Tensor], targeted = False) -> List[np.array]:

        target_tensor = torch.zeros((1000,1))[self.target,0] = 1
        target_tensor = [target_tensor for _ in range(5)]
        target_tensor = torch.reshape(target_tensor, (1000,5))

        pinit = np.random.uniform(-0.01, 0.01, size=image[0].shape)
        pertubation = torch.from_numpy(pinit)

        pertubation.requires_grad = True

        for it in range(self.steps):
            adv_image = image + pertubation
            pred = self.model(adv_image)
            loss = functional.CrossEntropyLoss(torch.max(pred).unsqueeze(0), pred)
            loss.backward()
            if it%10 == 0:
                print(f"iteration: {it}, loss: {loss.data.numpy()}")
            
            
            grads_sign = -1*torch.sign(pertubation.grad.data) if targeted else torch.sign(pertubation.grad.data)
            pertubation.add(grads_sign * self.epsilon)
            pertubation = torch.clamp(pertubation, -self.clip_value, self.clip_value)

        return pertubation.numpy()

