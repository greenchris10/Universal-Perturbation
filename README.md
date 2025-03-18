<div align="center">
    <h1> Targeted View-Invariant Adversarial Perturbations for 3D Object Recognition </h1>

  <a href='https://arxiv.org/abs/2412.13376'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a>
    <div>
        <a href='https://github.com/memoatwit' target='_blank'>M. Ergezer <sup>1, 2</sup></a>&emsp;
    <a href='https://github.com/greenchris10' target='_blank'>C. Green <sup>1</sup></a>&emsp;
          <a href='https://github.com/azeybey' target='_blank'> A. Zeybey <sup>1</sup></a>&emsp;
  </div>
  <br>
  <div>
      <sup>1</sup> Wentworth Institute of Technology <br>
      <sup>2</sup> Dr. Ergezer holds concurrent appointments as an Associate Professor at Wentworth Institute of Technology and as an Amazon Visiting Academic. This paper describes work performed at Wentworth Institute of Technology and is not associated with Amazon.
  </div>
</div>

## Abstract
Adversarial attacks pose significant challenges in 3D object recognition, especially in scenarios involving multi-view analysis where objects can be observed from varying angles. This paper introduces View-Invariant Adversarial Perturbations (VIAP), a novel method for crafting robust adversarial examples that remain effective across multiple viewpoints. Unlike traditional methods, VIAP enables targeted attacks capable of manipulating recognition systems to classify objects as specific, pre-determined labels, all while using a single universal perturbation. Leveraging a dataset of 1,210 images across 121 diverse rendered 3D objects, we demonstrate the effectiveness of VIAP in both targeted and untargeted settings. Our untargeted perturbations successfully generate a singular adversarial noise robust to 3D transformations, while targeted attacks achieve exceptional results, with top-1 accuracies exceeding 95% across various epsilon values. These findings highlight VIAPs potential for real-world applications, such as testing the robustness of 3D recognition systems. The proposed method sets a new benchmark for view-invariant adversarial robustness, advancing the field of adversarial machine learning for 3D object recognition. A sample implementation is made available [here](https://github.com/memoatwit/UniversalPerturbation).


## Article
The paper is presented at the AAAI-25 Workshop on Artificial Intelligence for Cyber Security (AICS) and a preprint is submitted to [arXiv](https://arxiv.org/abs/2412.13376) and can be cited as:
```
@misc{green2024targetedviewinvariantadversarialperturbations,
      title={Targeted View-Invariant Adversarial Perturbations for 3D Object Recognition}, 
      author={Christian Green and Mehmet Ergezer and Abdurrahman Zeybey},
      year={2024},
      eprint={2412.13376},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.13376}, 
}
```
