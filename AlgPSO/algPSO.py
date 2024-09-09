# -*- coding: utf-8 -*-
"""algPSO.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eqyyrtK14Fw6LmSzqDp3WNd6QOMXQbn9
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Função objetivo
def objective_function(X, n):
    t1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / n))
    t2 = -np.exp(np.sum(np.cos(2 * np.pi * X)) / n)
    return t1 + t2 + 20 + np.e

# Atualiza a velocidade da partícula
def update_velocity(V, X, Pbest, gbest, w, c1, c2, Vmin, Vmax):
    r1, r2 = np.random.rand(), np.random.rand()
    aux = (w * V +
           c1 * r1 * (Pbest - X) +
           c2 * r2 * (gbest - X))

    return np.clip(aux, Vmin, Vmax)

# Atualiza a posição da partícula
def update_position(X, V, Xmin, Xmax):
    aux = X + V

    return np.clip(aux, Xmin, Xmax)

# Inicializa as partículas
def initialize_particles(m, n, Xmin, Xmax, Vmin, Vmax):
    X = np.random.uniform(Xmin, Xmax, (m, n))
    V = np.random.uniform(Vmin, Vmax, (m, n))
    Pbest = X.copy()
    f_values = np.array([objective_function(p, n) for p in X])

    return X, V, Pbest, f_values

# Função principal
def PSO(Xmin, Xmax, Vmin, Vmax, c1, c2, k, m, n, w):
    X, V, Pbest, Pbest_f = initialize_particles(m, n, Xmin, Xmax, Vmin, Vmax)
    gbest = X[0]
    gbest_f = Pbest_f[0]
    history = []

    for i in range(k):
        history.append(X.copy())

        for j in range(m):
            V[j] = update_velocity(V[j], X[j], Pbest[j], gbest, w, c1, c2, Vmin, Vmax)
            X[j] = update_position(X[j], V[j], Xmin, Xmax)

            f = objective_function(X[j], n)
            if f < Pbest_f[j]:
                Pbest[j] = X[j]
                Pbest_f[j] = f

                if f < gbest_f:
                    gbest = X[j]
                    gbest_f = f

    return gbest, gbest_f, history

# Faz o GIF
def save_gif(history, Xmin, Xmax, n, filename='pso_evolution.gif'):
    if n > 3:
        print(f"Cannot plot for dimensions greater than 3.")
        return

    elif n == 3:
        with imageio.get_writer(filename, mode='I', duration=0.5) as writer:
            for iteration, positions in enumerate(history):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
                ax.set_xlim(Xmin, Xmax)
                ax.set_ylim(Xmin, Xmax)
                ax.set_zlim(Xmin, Xmax)
                ax.set_title(f'Iteration: {iteration+1}')
                ax.set_xlabel('Dimension 1')
                ax.set_ylabel('Dimension 2')
                ax.set_zlabel('Dimension 3')
                plt.grid(True)

                plt.savefig('temp.png')
                plt.close()

                image = imageio.imread('temp.png')
                writer.append_data(image)

    elif n == 2:
        with imageio.get_writer(filename, mode='I', duration=0.5) as writer:
            for iteration, positions in enumerate(history):
                plt.figure(figsize=(6, 6))
                plt.scatter(positions[:, 0], positions[:, 1])
                plt.xlim(Xmin, Xmax)
                plt.ylim(Xmin, Xmax)
                plt.title(f'Iteration: {iteration+1}')
                plt.xlabel('Dimension 1')
                plt.ylabel('Dimension 2')
                plt.grid(True)

                plt.savefig('temp.png')
                plt.close()

                image = imageio.imread('temp.png')
                writer.append_data(image)

Xmin, Xmax = -2.0, 2.0 # Limites para as posições
Vmin, Vmax = -1.0, 1.0 # Limites para as velocidades

c1 = 1.5 # Influência da melhor posição pessoal (Pbest)
c2 = 2.0 # Influência da melhor posição global (Gbest)
w = 0.7 # Fator de inércia

k = 200 # Número de iterações
m = 100 # Número de partículas no enxame
n = 3 # Dimensionalidade do problema (n variáveis por partícula)

gbest, gbest_f, history = PSO(Xmin, Xmax, Vmin, Vmax, c1, c2, k, m, n, w)
save_gif(history, Xmin, Xmax, n)

print(gbest, gbest_f)