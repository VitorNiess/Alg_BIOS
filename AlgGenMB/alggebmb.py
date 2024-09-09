# -*- coding: utf-8 -*-
"""AlgGebMB.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1e4tOAoBps3bgG74jR9-QsRGqWF9n_4at
"""

from random import *
from math import *
import matplotlib.pyplot as plt

def genRandBin(bits): #Gera binários aleatórios
  bins = []

  for i in range(bits):
    bins.append(randint(0, 1))

  return bins

def genIniPop(popSize, bits): #Gera população inicial, criando vários números aleatótios (popSize números)
  pop = []

  for i in range(popSize):
    pop.append(genRandBin(bits))

  return pop

def objFuncMin(pop, maxW, objW, objV): #Função objetivo com penalização mais branda
  sumW = 0
  sumV = 0
  calc = 0
  allSums = []

  for e in range(len(pop)):
    for i in range(len(pop[e])):

      if pop[e][i] == 1:
        sumW = sumW + objW[i]
        sumV = sumV + objV[i]

    if sumW > maxW:
      calc = sumV * (1 - (sumW - maxW) / maxW)

      allSums.append((calc, e))
    else:
      allSums.append((sumV, e))

    calc = 0
    sumW = 0
    sumV = 0

  return allSums

def objFuncMax(pop, maxW, objW, objV): #Função objetivo com penalização mais rígida
  sumW = 0
  sumV = 0
  calc = 0
  allSums = []

  for e in range(len(pop)):
    for i in range(len(pop[e])):

      if pop[e][i] == 1:
        sumW = sumW + objW[i]
        sumV = sumV + objV[i]

    if sumW > maxW:
      calc = sumV - (sumV * (sumW - maxW))

      allSums.append((calc, e))
    else:
      allSums.append((sumV, e))

    calc = 0
    sumW = 0
    sumV = 0

  return allSums

def roul(pop, fits, scalFactor): #Define a roleta e seleciona um pai para o cruzamento
    #Normaliza a aptidão
    minFit = min(fit[0] for fit in fits)
    maxFit = max(fit[0] for fit in fits)
    normalFit = [(fit[0] - minFit) / (maxFit - minFit) for fit in fits]

    #Aplica o escalonamento exponencial
    scaledFit = [fit ** scalFactor for fit in normalFit]

    #Calcula as probabilidades de seleção
    totalFit = sum(scaledFit)
    prob = [fit / totalFit for fit in scaledFit]

    #Roda da roleta (wheel)
    wheel = []
    accumulatedProb = 0

    for n in prob:
        accumulatedProb += n
        wheel.append(accumulatedProb)

    spin = random()
    selected = None

    for i, slice in enumerate(wheel):
        if spin <= slice:
            sel = pop[fits[i][1]]
            break

    return sel

def greater(ele1, ele2): #Verifica qual dos valores de fit é o maior

  if(ele1[0] > ele2[0]):
    return ele1
  else:
    return ele2

def tourn(fits): #Faz o torneio entre dois candidatos recebidos
  can1 = randint(0, (len(fits) - 1))
  can2 = randint(0, (len(fits) - 1))

  while(can1 == can2):
    can1 = randint(0, (len(fits) - 1))
    can2 = randint(0, (len(fits) - 1))

  return greater(fits[can1], fits[can2])

def defGroupSize(bits): #Define os cortes que serão feitos para o cruzamento
  gpSizes = []

  numGp1 = 0
  numGp2 = 0
  numGp3 = 0

  numGp1 = randint(1, (bits - 2)) #Define um tamanho aleatório para o primeiro grupo
  #O tamanho do grupo 1 deve ser menor que bits - 2 para que sobre pelo menos 1 elemento para os grupos 2 e 3

  if ((bits - 2) - numGp1) == 0: #Caso citado anteriormente, em que os grupos 2 e 3 podem ter apenas 1 elemento cada
    numGp2 = 1
  else: #Caso em que existem mais de 2 elementos disponíveis para os grupos 2 e 3
    numGp2 = randint(1, (bits - 2) - numGp1) #Define um tamanho aleatório para o grupo 2

  numGp3 = bits - (numGp1 + numGp2) #Define o tamanho do grupo 3 (os elementos restantes)

  gpSizes.append(numGp1)
  gpSizes.append(numGp2)
  gpSizes.append(numGp3)

  return gpSizes

def mutate(son, mutRate): #Faz a mutação de um indivíduo

  for i in range(len(son)):

    if(uniform(0, 1) <= mutRate): #Caso um número menor que a taxa seja sorteado, o gene é mutado
      if(son[i] == 0):
        son[i] = 1
      else:
        son[i] = 0

  return son

def crossOver(fat, mot, bits, mutRate): #Faz o cruzamento de dois pais individuais
  gpSize = defGroupSize(bits) #Define o tamanho dos grupos após os cortes

  son = []

  for i in range(bits): #Faz o cruzamento em si

    if(i <= gpSize[0]):
      son.append(mot[i])
    elif(i > gpSize[0] and i <= gpSize[1]):
      son.append(fat[i])
    else:
      son.append(mot[i])

  return mutate(son, mutRate) #Retorna o filho já mutado

def iniCross(pop, fits, bits, popSize, mutRate, selMet, scalFactor): #Faz o torneio, o cruzamento com os vencedores, a mutação com os filhos e retorna a próxima geração - 1 indíviduo (espaço reservado para o melhor da geração anterior)
  win = []

  if selMet == 0:
    for i in range(popSize): #Encontra os vencedores do torneio
      win.append(tourn(fits))
  elif selMet == 1:
    for i in range(popSize): #Encontra os selecionados na roleta
      win.append(roul(pop, fits, scalFactor))

  sons = []

  for i in range((popSize - 1)): #Faz o cruzamento entre os pais escolhidos
    sons.append(crossOver(frtGen[win[i][1]], frtGen[win[i + 1][1]], bits, mutRate))

  return sons

#Foi usado o primeiro exemplo dos arquivos disponibilizados

objW = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82] #Lista dos pesos dos objetos
objV = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72] #Lista dos valores dos objetos
maxW = 165 #Capacidade máxima da mochila

objFunc = 0 #Define qual função de penalização será utilizada (0 - branda, 1 - rígida)
selMet = 0 #Define qual método de seleção de pais será usada (0 - torneio, 1 - roleta)
scalFactor = 2 #Define o fator de escalonamento para o caso do método da roleta ser usado

bits = len(objW) #Quantidade de bits em cada indivíduo
popSize = 100 #Tamanho de cada população
genNum = 100 #Quantidade de gerações
mutRate = 0.1 #Taxa de mutação

frtGen = genIniPop(popSize, bits) #Gera a população inicial

aux = [] #Vetor auxiliar para a troca de gerações (é desnecessário, mas acho que didaticamente fica melhor)
fits = [] #Armazena os fits de cada geração
best = [] #Armazena os melhores valores para passar para a próxima geração
numFit = [] #Armazena os melhores fits para a análise final
numGen = [] #Armazena os números das gerações para a análise final

for i in range(genNum):

  if(objFunc == 0):
    fits = objFuncMin(frtGen, maxW, objW, objV) #Calcula os fits dos indivíduos da geração
  else:
    fits = objFuncMax(frtGen, maxW, objW, objV) #Calcula os fits dos indivíduos da geração

  aux = iniCross(frtGen, fits, bits, popSize, mutRate, selMet, scalFactor) #Retorna a nova geração após o cruzamento

  fits.sort(reverse=True)

  best = frtGen[fits[0][1]] #Armazena o melhor indivíduo da geração
  numFit.append(fits[0][0]) #Armazena o melhor fit da geração

  aux.append(best) #Adiciona o melhor da geração anterior na nova

  frtGen = aux
  aux = []
  fits = []

  numGen.append(i)

print("\n")
plt.plot(numGen, numFit)
plt.show()

numFit.sort(reverse=True)
print("\n\nMelhor geral:")
print(numFit[0])