# Análise de Algoritmos Genéticos para Otimização da Função F6

Este repositório contém o código-fonte e a análise de um trabalho sobre a aplicação de Algoritmos Genéticos (AGs) para encontrar o valor máximo da função de otimização $F6(x, y)$. O projeto explora tanto a implementação de um AG padrão quanto uma estratégia avançada de mutação adaptativa, com uma forte ênfase na visualização de dados para a análise de desempenho.

## 1. Descrição do Problema

O objetivo do trabalho é otimizar a função $F6(x, y)$, que é definida por:

$$F6(x, y) = 0.5 - \frac{\left[ \sin\left(\sqrt{x^2 + y^2}\right) \right]^2 - 0.5}{\left[ 1 + 0.001(x^2 + y^2) \right]^2}$$

- **Domínio:** As variáveis $x$ e $y$ estão no intervalo $[-100, +100]$.
- **Máximo Global:** O valor máximo da função é 1, que ocorre no ponto $(x, y) = (0, 0)$.

## 2. Metodologia Aplicada

A metodologia do projeto é dividida em duas partes principais:

### 2.1. Parte 1: Algoritmo Genético Padrão

Esta seção implementa um AG clássico, seguindo os seguintes princípios:
- **Codificação:** Utiliza uma representação binária, com 25 bits para cada variável ($x$ e $y$).
- **Parâmetros:**
    - Tamanho da População: 100
    - Taxa de Cruzamento: 80% (cruzamento de um ponto)
    - Taxa de Mutação: 1% (fixa)
- **Análise de Desempenho:** O algoritmo é avaliado monitorando a aptidão máxima, aptidão média e diversidade da população (medida pelo desvio padrão).

### 2.2. Parte 2: Mutação Adaptativa

Esta seção introduz uma estratégia avançada para a taxa de mutação:
- A taxa de mutação, inicialmente em 1%, é aumentada para 5% quando a aptidão máxima da população atinge um limiar de 0.99.
- O objetivo é demonstrar como essa estratégia pode reintroduzir diversidade na população após a convergência, permitindo uma análise mais profunda do comportamento do AG.


### Autora 

Amanda Gabrielly Prestes Lopes 202207040043