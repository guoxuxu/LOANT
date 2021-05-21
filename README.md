# LOANT
code for NAACL 2021 paper "Latent-Optimized Adversarial Neural Transfer for Sarcasm Detection"

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

<img src="./img/Gradient_Descent.png" width="250"/><img src="./img/First_Order.png" width="250"/><img src="./img/Second_Order.png" width="250"/>


## Sarcasm Detection Task
my working env, download: [compressed conda env](https://drive.google.com/file/d/1QMyHGEWrSIJ7eSN3kmsc6KOc6edfMOVN/view)

```mkdir -p my_env```

```tar -xzf my_env.tar.gz -C my_env```

```source my_env/bin/activate```

run experiments, e.g.:

```python main.py -source Ptacek -target iSarcasm -LO True```

## Sarcasm Datasets
1. Ghosh
2. Ptacek
3. SemEval18
4. iSarcasm
#### Processing Steps:
1. detect language, filter out non-english tweets
2. lexical normalization
3. filter out duplicate tweets across datasets
4. up-sampling target datasets