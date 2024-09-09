# -*- coding: utf-8 -*-
"""AlgGen.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gVxW1FbCrMJoxDWS37efayf7Yw7ofVea
"""

from random import *
from math import *
import matplotlib.pyplot as plt

def binToInt(bin): #Função que converte um decimal em binário
  dec = 0
  tam = len(bin)

  for x in range(tam):
    dec += (pow(2, x) * bin[(tam - 1) - x])

  return dec

def discre(dec, bits): #Fórmula para discretizar os valores
  min = -2
  max = 2

  return min + ((max - min) / (pow(2, bits) - 1)) * dec

def sum1(decs): #Primeiro somatório da função objetivo
  res = 0

  for x in decs:
    res = res + pow(x, 2)

  return res

def sum2(decs): #Segundo somatório da função objetivo
  res = 0

  for x in decs:
    res = res + cos(2 * pi * x)

  return res

def obFunc(bins, varNum, bits): #Função objetivo
  ints = []
  intsDisc = []

  for i in range(varNum): #Converte os 3 binários recebidos para decimal
    ints.append(binToInt(bins[i]))

  for x in ints: #Discretiza os 3 valores decimais
    intsDisc.append(discre(x, bits))

  return (-20 * pow(e, (-0.2 * sqrt((1 / varNum) * sum1(intsDisc))))) - (pow(e, ((1 / varNum) * sum2(intsDisc)))) + 20 + e

def genRandBin(varNum, bits): #Gera binários aleatórios, com base na precisão da função e no número máximo de bits
  bins = []
  binAux = []

  for i in range(varNum):
    for j in range(bits):
      binAux.append(randint(0, 1))

    bins.append(binAux)
    binAux = []

  return bins

def genIniPop(popSize, varNum, bits): #Gera população inicial, criando vários números aleatótios (popSize números)
  pop = []

  for i in range(popSize):
    pop.append(genRandBin(varNum, bits))

  return pop

def calcObFunc(bins, varNum, bits, pos): #Calcula o valor da função objetivo individualmente

  return (obFunc(bins, varNum, bits), pos)

def less(ele1, ele2): #Verifica qual dos valores de fit é o menor

  if(ele1[0] < ele2[0]):
    return ele1
  else:
    return ele2

def tourn(fitsGen, popSize): #Faz o torneio entre dois candidatos recebidos
  lesser = []
  can1 = randint(0, (popSize - 1))
  can2 = randint(0, (popSize - 1))

  while(can1 == can2):
    can1 = randint(0, (popSize - 1))
    can2 = randint(0, (popSize - 1))

  lesser = less(fitsGen[can1], fitsGen[can2])

  return lesser

def defGroupSize(bits, varNum): #Define os cortes que serão feitos para o cruzamento
  gpSizes = []

  numGp1 = 0
  numGp2 = 0
  numGp3 = 0

  numGp1 = randint(1, ((bits * varNum) - 2)) #Define um tamanho aleatório para o primeiro grupo
  #O tamanho do grupo 1 deve ser menor que (bits * varNum) - 2 para que sobre pelo menos 1 elemento para os grupos 2 e 3

  if (((bits * varNum) - 2) - numGp1) == 0: #Caso citado anteriormente, em que os grupos 2 e 3 podem ter apenas 1 elemento cada
    numGp2 = 1
  else: #Caso em que existem mais de 2 elementos disponíveis para os grupos 2 e 3
    numGp2 = randint(1, ((bits * varNum) - 2) - numGp1) #Define um tamanho aleatório para o grupo 2

  numGp3 = (bits * varNum) - (numGp1 + numGp2) #Define o tamanho do grupo 3 (os elemntos restantes)

  gpSizes.append(numGp1)
  gpSizes.append(numGp2)
  gpSizes.append(numGp3)

  return gpSizes

def crossOver(fat, mot, bits, varNum): #Faz o cruzamento de dois pais individuais
  aux1 = []
  aux2 = []

  for i in range(varNum): #Necessário mudar o formato dos pais para fazer os cortes (de [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] para [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for j in range(bits):
      aux1.append(fat[i][j])

  for i in range(varNum): #Necessário mudar o formato dos pais para fazer os cortes (de [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] para [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for j in range(bits):
      aux2.append(mot[i][j])

  gpSize = defGroupSize(bits, varNum) #Define o tamanho dos grupos após os cortes

  son1 = []

  for i in range((bits * varNum)): #Faz o cruzamento em si

    if(i <= gpSize[0]):
      son1.append(aux2[i])
    elif(i > gpSize[0] and i <= gpSize[1]):
      son1.append(aux1[i])
    else:
      son1.append(aux2[i])

  return son1

def chanForm(son, bits, varNum, mutRate): #Converte a forma do indivíduo (de [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] para [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
  aux = []
  reSon = []
  x = 0

  for i in range(varNum):
    for j in range(bits):
      aux.append(son[x])
      x += 1

    reSon.append(aux)
    aux = []

  return mutate(reSon, mutRate)

def mutate(son, mutRate): #Faz a mutação de um indivíduo

  for i in range(len(son)):

    for j in range(len(son[i])):

      if(uniform(0, 1) <= mutRate): #Caso um número menor que a taxa seja sorteado, o gene é mutado
        if(son[i][j] == 0):
          son[i][j] = 1
        else:
          son[i][j] = 0

  return son

def iniCross(frtGen, fitsGen, bits, varNum, popSize, mutRate): #Faz o torneio, o cruzamento com os vencedores, a mutação com os filhos e retorna a próxima geração - 1 indíviduo (espaço reservado para o melhor da geração anterior)
  win = []

  for i in range(popSize): #Encontra os vencedores do torneio
    win.append(tourn(fitsGen, popSize))

  sons = []
  nxtGen = []

  for i in range((popSize - 1)): #Faz o cruzamento entre os vencedores
    sons.append(crossOver(frtGen[win[i][1]], frtGen[win[i + 1][1]], bits, varNum))

  for i in range(len(sons)): #Altera a forma dos filhos para a operável pelas funções (de [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] para [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    nxtGen.append(chanForm(sons[i], bits, varNum, mutRate))

  return nxtGen

varNum = 3 #Quantidade de números a serem testados (precisão da função)
bits = 6 #Número de bits máximo
popSize = 10 #Tamanho de cada população
genNum = 100 #Quantidade de gerações
mutRate = 0.05 #Taxa de mutação

frtGen = genIniPop(popSize, varNum, bits) #Gera a população inicial

aux = [] #Vetor auxiliar para a troca de gerações (é desnecessário, mas acho que didaticamente fica melhor)
fitsGen = [] #Armazena os fits de cada geração
best = [] #Armazena os melhores valores para passar para a próxima geração
numFit = [] #Armazena os melhores fits para a análise final
numGen = [] #Armazena os números das gerações para a análise final

for i in range(genNum):

  for j in range(popSize):
    fitsGen.append(calcObFunc(frtGen[j], varNum, bits, j)) #Calcula os fits dos indivíduos da geração

  aux = iniCross(frtGen, fitsGen, bits, varNum, popSize, mutRate) #Retorna a nova geração após o cruzamento

  fitsGen.sort()

  best = frtGen[fitsGen[0][1]] #Armazena o melhor indivíduo da geração
  numFit.append(fitsGen[0][0]) #Armazena o melhor fit da geração

  aux.append(best) #Adiciona o melhor da geração anterior na nova

  frtGen = aux
  aux = []

  print("\nTrês melhores fits da geração:")
  for j in range(3):
    print(fitsGen[j][0])

  fitsGen = []

  numGen.append(i)

print("\n")
plt.plot(numGen, numFit)
plt.show()

numFit.sort()
print("\n\nMelhor geral:")
print(numFit[0])