import math
import random
import copy
import matplotlib.pyplot as plt
import numpy as np

def readCts(file): # Lê as cidades do arquivo
    with open(file, 'r') as f:
        lines = f.readlines()

    cNum = len(lines)
    disM = [list(map(float, line.strip().split())) for line in lines]

    return cNum, disM

def genIniPop(popSize, cNum): # Gera a população inicial
    pop = [random.sample(range(cNum), cNum) for _ in range(popSize)]

    return pop

def fitness(pop, disM): # Calcula os fitness
    fit = []

    for route in pop:
        dist = sum(disM[route[i]][route[i + 1]] for i in range(len(route) - 1))
        dist += disM[route[-1]][route[0]]
        fit.append(dist)

    return fit

def tourn(popSize, fit): # Método do torneio
    fat = []
    vicP = 0.9

    for _ in range(popSize):
        can1, can2 = random.sample(range(popSize), 2)
        vencedor = can1 if fit[can1] < fit[can2] else can2

        if random.random() > vicP:
            vencedor = can1 if vencedor == can2 else can2

        fat.append(vencedor)

    return fat

def roul(popSize, fit): # Método da roleta (acredito estar errado)
    total_fit = sum(1 / f for f in fit)
    prob = [(1 / f) / total_fit for f in fit]

    cumulative_sum = np.cumsum(prob)

    fat = [np.searchsorted(cumulative_sum, random.random()) for _ in range(popSize)]

    return fat

def crossover(pop, fat, cNum, crossRate): # Faz o cruzamento
    newPop = []

    for i in range(0, len(fat) - 1, 2):

        if random.random() <= crossRate:
            parent1, parent2 = pop[fat[i]], pop[fat[i + 1]]

            start, end = sorted(random.sample(range(cNum), 2))

            child1, child2 = parent1[start:end], parent2[start:end]

            fill1 = [item for item in parent2 if item not in child1]
            fill2 = [item for item in parent1 if item not in child2]

            newPop.append(fill1[:start] + child1 + fill1[start:])
            newPop.append(fill2[:start] + child2 + fill2[start:])
        else:
            newPop.append(pop[fat[i]])
            newPop.append(pop[fat[i + 1]])

    return newPop

def mutate(pop, cNum, mutRate): # Faz a mutação

    for i in range(len(pop)):
        if random.random() <= mutRate:
            a, b = random.sample(range(cNum), 2)
            pop[i][a], pop[i][b] = pop[i][b], pop[i][a]

    return pop

def elitismo(pop, fit): # Faz o elitismo
    return pop[np.argmin(fit)]

def plota(resultados, genNum): # Plota o gráfico final
    geracoes = list(range(genNum))

    plt.plot(geracoes, resultados, label="Melhor Resultado", linestyle='-', marker='.', color='blue')
    plt.title('Algoritmo Genético')
    plt.xlabel('Gerações')
    plt.ylabel('Melhor resultado')
    plt.grid(True)
    plt.legend()
    plt.show()

def execAlgGen(popSize, genNum, cNum, pop, disM, selMet, mutRate, crossRate): # Executa o algoritmo em si
    resultados = []

    for _ in range(genNum):
        fit = fitness(pop, disM) # Calcula o fitness
        fat = tourn(popSize, fit) if selMet == 0 else roul(popSize, fit) # Faz a seleção de pais, de acordo com o método escolhido
        pop = crossover(pop, fat, cNum, crossRate) # Faz o cruzamento
        pop = mutate(pop, cNum, mutRate) # Faz a mutação
        best_individual = elitismo(pop, fit) # Seleciona o melhor fit para manter o elitismo
        resultados.append(min(fit)) # Salva o melhor para a análise
        pop[-1] = best_individual  # Mantém o elitismo

    # Salva os resultados no arquivo, ordenados de forma decrescente
    with open("saida_1.txt", "w") as arquivo:
        for resultado in sorted(resultados, reverse=True):
            arquivo.write(f"{resultado}\n")

    plota(resultados, genNum)

popSize = 200 # Tamanho da população
genNum = 500 # Número de gerações
mutRate = 0.01 # Taxa de mutação
crossRate = 1 # Taxa de cruzamento
selMet = 0 # Seleciona o método de cruzamento (0 - torneio e 1 - roleta)

cNum, disM = readCts("lau15_dist.txt") # Lê as cidades do arquivo (usei o lau15_dist.txt)

pop = genIniPop(popSize, cNum)
execAlgGen(popSize, genNum, cNum, pop, disM, selMet, mutRate, crossRate)