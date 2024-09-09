# -*- coding: utf-8 -*-
"""Alg_WOA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rNvsT_SfGB5HBwYwIcVfKo3_cBu7xHx6
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

class Baleia:
    def __init__(self, Xmin, Xmax, n, func_espiral):
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.n = n
        self.func_espiral = func_espiral  # Armazena a função de espiral para cada baleia

        self.X = np.random.uniform(self.Xmin, self.Xmax, n)
        self.fitness = self.funcao_objetivo()

    def funcao_objetivo(self):
        t1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(self.X**2) / self.n))
        t2 = -np.exp(np.sum(np.cos(2 * np.pi * self.X)) / self.n)
        return t1 + t2 + 20 + np.e

    def cercamento(self, melhor_baleia, A, D):
        self.X = melhor_baleia - A * D

    def espiral_de_bolhas(self, melhor_baleia, D):
        # Passa a posição da baleia e D para a função de espiral
        self.X = self.func_espiral(self.X, melhor_baleia, D)

    def buscar_novas_presas(self, Xrand, A, D):
        self.X = Xrand - A * D

    def atualizar_posicao(self, melhor_baleia, A, C, p, Xrand=None):
        D = np.abs(C * melhor_baleia - self.X)
        if p < 0.5:
            if np.all(np.abs(A) < 1):
                self.cercamento(melhor_baleia, A, D)
            else:
                if Xrand is not None:
                    self.buscar_novas_presas(Xrand, A, D)
        else:
            self.espiral_de_bolhas(melhor_baleia, D)

        self.X = np.clip(self.X, self.Xmin, self.Xmax)
        self.f = self.funcao_objetivo()

class Baleal:
    def __init__(self, Xmin, Xmax, num_iter, num_baleia, n):
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.num_iter = num_iter
        self.num_baleia = num_baleia
        self.n = n

        self.baleias = []
        self.history = []
        self.melhor_baleia = None
        self.melhor_f = None

    def inicializar(self, funcoes_espiral):
        num_func = len(funcoes_espiral)
        for i in range(self.num_baleia):
            func_espiral = funcoes_espiral[i % num_func]  # Cicla entre as funções de espiral
            baleia = Baleia(self.Xmin, self.Xmax, self.n, func_espiral)
            self.baleias.append(baleia)
        self.melhor_baleia = self.baleias[0].X
        self.melhor_f = self.baleias[0].funcao_objetivo()

    def WOA(self, funcoes_espiral):
        self.inicializar(funcoes_espiral)

        for i in range(self.num_iter):
            a = 2 * (1 - np.log(i + 1) / np.log(self.num_iter + 1))  # Decaimento de a
            for baleia in self.baleias:
                r = np.random.uniform(0, 1, self.n)
                A = 2*a * r - a
                C = 2 * r
                p = np.random.rand()
                Xrand = self.baleias[np.random.randint(self.num_baleia)].X

                baleia.atualizar_posicao(self.melhor_baleia, A, C, p, Xrand)

            for baleia in self.baleias:
                if baleia.f < self.melhor_f:
                    self.melhor_baleia = baleia.X
                    self.melhor_f = baleia.f

            current_positions = np.array([baleia.X for baleia in self.baleias])
            self.history.append(current_positions)

        return self.melhor_baleia, self.melhor_f

    def save_gif(self, filename="woa_evolution.gif"):
        colors = ['red', 'blue', 'green', 'purple']  # Cores para cada função
        with imageio.get_writer(filename, mode='I', duration=400) as writer: # Duration define a duração do gif (quanto maior mais demorado)
            cont = 0
            for positions in self.history:
                plt.figure(figsize=(6, 6))
                num_baleias_por_funcao = len(self.baleias) // len(colors)

                for idx, color in enumerate(colors):
                    start = idx * num_baleias_por_funcao
                    end = start + num_baleias_por_funcao
                    plt.scatter(positions[start:end, 0], positions[start:end, 1], color=color, label=f'Função {idx+1}')

                plt.xlim(-3, 3)
                plt.ylim(-3, 3)
                plt.title(f'Iteração: {cont + 1}')  # Atualizando cont + 1
                plt.xlabel('Dimensão 1')
                plt.ylabel('Dimensão 2')
                plt.grid(True)
                plt.legend()

                plt.savefig('temp.png')
                plt.close()

                image = imageio.imread('temp.png')
                writer.append_data(image)
                cont += 1

# Definindo as quatro funções de espiral de bolhas
def espiral_1(X, melhor_baleia, D):
    l = np.random.uniform(-1, 1)
    return D * np.exp(l) * np.sin(2 * np.pi * l) + melhor_baleia

def espiral_2(X, melhor_baleia, D):
    l = np.random.uniform(-1, 1)
    return D * np.exp(l) * np.cos(2 * np.pi * l) * np.log(1 + np.abs(l)) + melhor_baleia

def espiral_3(X, melhor_baleia, D):
    l = np.random.uniform(-1, 1)
    scale = np.array([1.5, 0.5])
    return D * np.exp(l) * np.cos(2 * np.pi * l) * scale + melhor_baleia

def espiral_4(X, melhor_baleia, D):
    l = np.random.uniform(-1, 1)
    damping_factor = 0.1
    return D * np.exp(l) * np.cos(2 * np.pi * l) * np.exp(-damping_factor * np.abs(l)) + melhor_baleia

# Executando o algoritmo com as diferentes funções de espiral
B = Baleal(-2.0, 2.0, 25, 500, 2) # X mínimo, X máximo, número de iterações, número de baleias, número de dimensões
funcoes_espiral = [espiral_1, espiral_2, espiral_3, espiral_4]
melhor_b, melhor_f = B.WOA(funcoes_espiral)
print(melhor_b, melhor_f)
B.save_gif()