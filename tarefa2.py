import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from mpl_toolkits.mplot3d import Axes3D

# Define a semente para garantir resultados reproduzíveis
np.random.seed(487)

# =================================================================================
# 1. PARÂMETROS CONFIGURÁVEIS
# =================================================================================
DOMAIN_MIN = -100
DOMAIN_MAX = 100
N_BITS = 25    
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.8
MAX_GENERATIONS = 500

# Parâmetros de Mutação Adaptativa
INITIAL_MUTATION_RATE = 0.01
ADAPTIVE_MUTATION_RATE_HIGH = 0.05   # Nova taxa de mutação para exploração
APTITUDE_THRESHOLD_HIGH = 0.99       # Limite de aptidão para acionar a exploração

# Parâmetros do GIF
GIF_FILENAME = 'ag_evolution.gif'
GIF_FPS = 10  # Quadros por segundo (Frames per second)
TEMP_DIR = 'temp_plots' # Diretório temporário para salvar os quadros

# =================================================================================
# 2. FUNÇÃO DE APTIDÃO (FITNESS FUNCTION)
# =================================================================================
def fitness_function(x, y):
    """
    Função F6(x, y) a ser maximizada.
    O máximo global é F6(0, 0) = 1.
    """
    try:
        numerator = (np.sin(np.sqrt(x**2 + y**2)))**2 - 0.5
        denominator = (1 + 0.001 * (x**2 + y**2))**2
        return np.where(denominator != 0, 0.5 - (numerator / denominator), -np.inf)
    except ZeroDivisionError:
        return -np.inf

# =================================================================================
# 3. FUNÇÕES DE CODIFICAÇÃO/DECODIFICAÇÃO
# =================================================================================
def decode_chromosome(chromosome):
    half_size = len(chromosome) // 2
    x_bits = chromosome[:half_size]
    y_bits = chromosome[half_size:]

    x_int = int("".join(str(b) for b in x_bits), 2)
    y_int = int("".join(str(b) for b in y_bits), 2)

    x = DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * x_int / (2**N_BITS - 1)
    y = DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * y_int / (2**N_BITS - 1)

    return x, y

# =================================================================================
# 4. OPERADORES DO ALGORITMO GENÉTICO
# =================================================================================
def create_population():
    return [np.random.randint(0, 2, size=2 * N_BITS).tolist() for _ in range(POPULATION_SIZE)]

def selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness <= 0:
        return np.random.choice(range(POPULATION_SIZE), size=int(POPULATION_SIZE * CROSSOVER_RATE), replace=True)

    probabilities = [f / total_fitness for f in fitnesses]
    parent_indices = np.random.choice(
        range(POPULATION_SIZE), 
        size=int(POPULATION_SIZE * CROSSOVER_RATE), 
        p=probabilities,
        replace=True
    )
    return parent_indices

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(chromosome, rate):
    for i in range(len(chromosome)):
        if np.random.rand() < rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def elitism(population, fitnesses):
    best_index = np.argmax(fitnesses)
    return population[best_index]

# =================================================================================
# 5. FUNÇÃO PRINCIPAL QUE EXECUTA TUDO
# =================================================================================
def run_full_analysis():
    
    # ----------------------
    # Fase 1: Execução do AG e Geração de dados e GIF
    # ----------------------
    population = create_population()
    best_chromosome = None
    best_fitness = -np.inf
    
    current_mutation_rate = INITIAL_MUTATION_RATE
    mutation_applied_generation = None
    
    max_fitness_per_generation = []
    avg_fitness_per_generation = []
    diversity_per_generation = []
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    print("Iniciando a execução do Algoritmo Genético e gerando quadros para o GIF...")
    for generation in range(1, MAX_GENERATIONS + 1):
        fitnesses = []
        x_coords, y_coords = [], []
        
        for chromosome in population:
            x, y = decode_chromosome(chromosome)
            fitness = fitness_function(x, y)
            fitnesses.append(fitness)
            x_coords.append(x)
            y_coords.append(y)
        
        fitnesses_np = np.array(fitnesses)
        current_diversity = np.std(fitnesses_np)
        
        max_fitness_per_generation.append(np.max(fitnesses_np))
        avg_fitness_per_generation.append(np.mean(fitnesses_np))
        diversity_per_generation.append(current_diversity)

        # Geração do GIF
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(x_coords, y_coords, c=fitnesses, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Geração {generation} - Melhor Aptidão: {np.max(fitnesses_np):.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
        ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        ax.grid(True)
        
        cbar = fig.colorbar(scatter, ax=ax, label='Aptidão')
        
        filename = f"{TEMP_DIR}/gen_{generation:04d}.png"
        plt.savefig(filename)
        plt.close(fig)

        current_best_fitness = max(fitnesses)
        if current_best_fitness >= APTITUDE_THRESHOLD_HIGH and mutation_applied_generation is None:
            current_mutation_rate = ADAPTIVE_MUTATION_RATE_HIGH
            mutation_applied_generation = generation
            print(f"Geração {generation}: Melhor aptidão atingiu {APTITUDE_THRESHOLD_HIGH:.2f}. Aumentando taxa de mutação para {ADAPTIVE_MUTATION_RATE_HIGH:.2f}")

        current_best_index = np.argmax(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = population[current_best_index]
            
        parent_indices = selection(population, fitnesses)
        
        next_population = []
        next_population.append(elitism(population, fitnesses))
        
        num_crossovers = int(POPULATION_SIZE * CROSSOVER_RATE)
        if num_crossovers % 2 != 0:
            num_crossovers += 1
        num_pairs = num_crossovers // 2

        for _ in range(num_pairs):
            p1_idx, p2_idx = np.random.choice(parent_indices, size=2, replace=False)
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, current_mutation_rate)
            child2 = mutation(child2, current_mutation_rate)
            next_population.extend([child1, child2])
        
        while len(next_population) < POPULATION_SIZE:
             next_population.append(np.random.randint(0, 2, size=2 * N_BITS).tolist())

        population = next_population[:POPULATION_SIZE]

    # Criação do GIF
    print("\nCombinando quadros para gerar o GIF...")
    filenames = sorted([f"{TEMP_DIR}/{f}" for f in os.listdir(TEMP_DIR) if f.endswith('.png')])
    
    with imageio.get_writer(GIF_FILENAME, mode='I', fps=GIF_FPS) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)
    
    os.rmdir(TEMP_DIR)
    
    final_x, final_y = decode_chromosome(best_chromosome)
    
    print("\n----------------------------------------------------")
    print("Algoritmo Genético Finalizado.")
    print(f"Melhor aptidão encontrada: {best_fitness:.6f}")
    print(f"Valores de (x, y) que geram essa aptidão: ({final_x:.6f}, {final_y:.6f})")
    print(f"Geração da mutação adaptativa: {mutation_applied_generation}")
    print(f"O GIF foi salvo como '{GIF_FILENAME}'")
    print("----------------------------------------------------")

    # ----------------------
    # Fase 2: Plotagem dos gráficos finais
    # ----------------------
    
    generations = range(1, MAX_GENERATIONS + 1)
    
    max_fitness_np = np.array(max_fitness_per_generation)
    avg_fitness_np = np.array(avg_fitness_per_generation)
    diversity_np = np.array(diversity_per_generation)
    
    # Normalização dos dados para plotagem
    normalized_max_fitness = (max_fitness_np - np.min(max_fitness_np)) / (np.max(max_fitness_np) - np.min(max_fitness_np))
    normalized_diversity = (diversity_np - np.min(diversity_np)) / (np.max(diversity_np) - np.min(diversity_np))
    normalized_avg_fitness = (avg_fitness_np - np.min(max_fitness_np)) / (np.max(max_fitness_np) - np.min(max_fitness_np))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Gráfico 1: Visão Geral ---
    ax1.plot(generations, normalized_max_fitness, 'k-', linewidth=2, label='Aptidão Máxima Normalizada')
    ax1.plot(generations, normalized_diversity, 'r-', linewidth=2, label='Diversidade (Desvio Padrão Normalizado)')
    ax1.plot(generations, normalized_avg_fitness, 'g--', linewidth=1, label='Aptidão Média Normalizada')
    
    if mutation_applied_generation:
        ax1.axvline(x=mutation_applied_generation, color='red', linestyle='--', label=f'Taxa de Mutação Aumentada (Geração {mutation_applied_generation})')

    ax1.set_title('Evolução Geral da Aptidão e Diversidade')
    ax1.set_xlabel('Gerações')
    ax1.set_ylabel('Valor Normalizado')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # --- Gráfico 2: Zoom na mudança ---
    if mutation_applied_generation:
        zoom_start = max(1, mutation_applied_generation - 10)
        zoom_end = min(MAX_GENERATIONS, mutation_applied_generation + 20)
        
        zoom_generations = generations[zoom_start-1:zoom_end]
        zoom_max_fitness = normalized_max_fitness[zoom_start-1:zoom_end]
        zoom_diversity = normalized_diversity[zoom_start-1:zoom_end]
        zoom_avg_fitness = normalized_avg_fitness[zoom_start-1:zoom_end]
        
        ax2.plot(zoom_generations, zoom_max_fitness, 'k-', linewidth=2, label='Aptidão Máxima Normalizada')
        ax2.plot(zoom_generations, zoom_diversity, 'r-', linewidth=2, label='Diversidade (Desvio Padrão Normalizado)')
        ax2.plot(zoom_generations, zoom_avg_fitness, 'g--', linewidth=1, label='Aptidão Média Normalizada')
        
        ax2.axvline(x=mutation_applied_generation, color='red', linestyle='--', label=f'Taxa de Mutação Aumentada (Geração {mutation_applied_generation})')

        ax2.set_title('Zoom na Mudança da Taxa de Mutação')
        ax2.set_xlabel('Gerações')
        ax2.set_ylabel('Valor Normalizado')
        ax2.legend(loc='upper right')
        ax2.grid(True)
    else:
        ax2.set_title("O zoom não pode ser exibido. Aptidão máxima não alcançada.")
        ax2.set_xlabel('Gerações')
        ax2.set_ylabel('Valor Normalizado')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# =================================================================================
# EXECUÇÃO DO SCRIPT
# =================================================================================
if __name__ == "__main__":
    run_full_analysis()