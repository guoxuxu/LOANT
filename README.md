# LOANT
code for NAACL 2021 paper "Latent-Optimized Adversarial Neural Transfer for Sarcasm Detection"

## Environment for reproduction
#### Steps:
My conda version is 4.9.2
1. Conda create yourenv
2. conda activate yourenv
3. install pytorch 1.6.1
4. pip install -r requirements.txt

## Toy exmpale for optimizing the 2D convex problem ![](http://www.sciweavers.org/tex2img.php?eq=f(w)=w^{T}Aw%2Bb^{T}w%2Bc&bc=White&fc=Black&im=jpg&fs=12) with extragradient.

#### Steps:
1. manually create a matrix with condition number of 40 (to control the loss landscape to be eclipse): ![](http://www.sciweavers.org/tex2img.php?eq=\Lambda=[[40,0],[0,1]]&bc=White&fc=Black&im=jpg&fs=12).
2. then generate a semi-definite matrix ![](http://www.sciweavers.org/tex2img.php?eq=A\in\mathbb{R}^{2\times2}&bc=White&fc=Black&im=jpg&fs=12}) (![](http://www.sciweavers.org/tex2img.php?eq=A=Q\LambdaQ^{T}&bc=White&fc=Black&im=jpg&fs=12)), a column vector b and a scalar c.
3. create a grid of ![](http://www.sciweavers.org/tex2img.php?eq=(w_0,w_1)&bc=White&fc=Black&im=jpg&fs=12).
4. generate the loss value ![](http://www.sciweavers.org/tex2img.php?eq=f(w)&bc=White&fc=Black&im=jpg&fs=12) and plot the loss contour.
5. perform **_gradient descent_** with initial point ![](http://www.sciweavers.org/tex2img.php?eq=(w_0=0,w_1=-0.15)&bc=White&fc=Black&im=jpg&fs=12) and learning rate ![](http://www.sciweavers.org/tex2img.php?eq=\eta=0.025&bc=White&fc=Black&im=jpg&fs=12), plot the trajectory of w.
6. perform **_first-order extragradient_** with the same initial point and learning rate, plot the trajectory of w.
7. perform **_Full hessian extragradient_** with the same initial point and learning rate, plot the trajectory of w.

#### Run
```python extragradient.py```