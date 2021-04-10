# LOANT
code for NAACL 2021 paper "Latent-Optimized Adversarial Neural Transfer for Sarcasm Detection"

## Environment for reproduction
#### Steps:
My conda version is 4.9.2
1. Conda create yourenv
2. conda activate yourenv
3. install pytorch 1.6.1
4. pip install -r requirements.txt

## Toy exmpale for minimizing the 2D function ![](https://render.githubusercontent.com/render/math?math=f(w)=w^{T}Aw%2Bb^{T}w%2Bc) with extragradient.

#### Steps:
1. manually create a matrix with condition number of 40 (to control the loss landscape to be eclipse): ![](https://render.githubusercontent.com/render/math?math=\Lambda=[[40,0],[0,1]]).
2. then generate a semi-definite matrix ![](https://render.githubusercontent.com/render/math?math=A\in\mathbb{R}^{2\times2}) (![](https://render.githubusercontent.com/render/math?math=A=Q\Lambda%20Q^{T})), a column vector b and a scalar c.
3. create a grid of ![](https://render.githubusercontent.com/render/math?math=(w_0,w_1)).
4. generate the loss value ![](https://render.githubusercontent.com/render/math?math=f(w)) and plot the loss contour.
5. perform **_gradient descent_** with initial point ![](https://render.githubusercontent.com/render/math?math=(w_0=0,w_1=-0.15)) and learning rate ![](https://render.githubusercontent.com/render/math?math=\eta=0.025), plot the trajectory of w.
6. perform **_first-order extragradient_** with the same initial point and learning rate, plot the trajectory of w.
7. perform **_Full hessian extragradient_** with the same initial point and learning rate, plot the trajectory of w.

#### Run
```python extragradient.py```

<!--<img src="./img/First_Order.png" width="200"> aaaaaa  | <img src="./img/First_Order.png" width="200"> aaaaaa -->
![Vanilla gradient descent](./img/First_Order.png){:height="36px" width="36px"}

[<img src="./img/First_Order.png" width="250"/>](image.png)

## Sarcasm Datasets pre-processing
#### Source:
1. Ghosh
2. Ptacek
3. SemEval18
4. iSarcasm
#### Steps:
