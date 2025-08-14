import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# =================================================================================
# 1. PARÂMETROS CONFIGURÁVEIS
# =================================================================================
DOMAIN_MIN = -100
DOMAIN_MAX = 100
N_BITS = 25    

POPULATION_SIZE = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
MAX_GENERATIONS = 500

# Parâmetros do GIF
GIF_FILENAME = 'ag_evolution.gif'
GIF_FPS = 10  # Quadros por segundo (Frames per second)
TEMP_DIR = 'temp_plots' # Diretório temporário para salvar os quadros

# =================================================================================
# 2. FUNÇÃO DE APTIDÃO (FITNESS FUNCTION)
# =================================================================================
def fitness_function(x, y):
    try:
        numerator = (np.sin(np.sqrt(x**2 + y**2)))**2 - 0.5
        denominator = (1 + 0.001 * (x**2 + y**2))**2
        return 0.5 - (numerator / denominator)
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

def mutation(chromosome):
    for i in range(len(chromosome)):
        if np.random.rand() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def elitism(population, fitnesses):
    best_index = np.argmax(fitnesses)
    return population[best_index]

# =================================================================================
# 5. LOOP PRINCIPAL E GERAÇÃO DO GIF
# =================================================================================
def genetic_algorithm_gif_plot():
    
    population = create_population()
    best_chromosome = None
    best_fitness = -np.inf
    
    # Adicionando a lista para armazenar o fitness máximo
    max_fitness_per_generation = []
    
    # Cria o diretório temporário para os plots
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    for generation in range(1, MAX_GENERATIONS + 1):
        fitnesses = []
        x_coords, y_coords = [], []
        
        for chromosome in population:
            x, y = decode_chromosome(chromosome)
            fitness = fitness_function(x, y)
            fitnesses.append(fitness)
            x_coords.append(x)
            y_coords.append(y)
        
        current_best_fitness = max(fitnesses)
        
        # Armazena o melhor fitness da geração para o gráfico final
        max_fitness_per_generation.append(current_best_fitness)
        
        # print(f"Geração {generation:03d}: Melhor aptidão = {current_best_fitness:.6f}")
        
        # --- Geração do plot e salvamento em arquivo ---
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(x_coords, y_coords, c=fitnesses, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Geração {generation} - Melhor Aptidão: {current_best_fitness:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
        ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
        ax.grid(True)
        
        cbar = fig.colorbar(scatter, ax=ax, label='Aptidão')
        
        filename = f"{TEMP_DIR}/gen_{generation:04d}.png"
        plt.savefig(filename)
        plt.close(fig)

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
            child1 = mutation(child1)
            child2 = mutation(child2)
            next_population.extend([child1, child2])
        
        while len(next_population) < POPULATION_SIZE:
             next_population.append(np.random.randint(0, 2, size=2 * N_BITS).tolist())

        population = next_population[:POPULATION_SIZE]

    # --- Criação do GIF a partir dos quadros salvos ---
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
    print(f"O GIF foi salvo como '{GIF_FILENAME}'")
    print("----------------------------------------------------")

    # Plot the fitness evolution graph
    generations = range(1, MAX_GENERATIONS + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_fitness_per_generation, label='Maximum Fitness')
    plt.axhline(y=1, color='r', linestyle='--', label='Global Maximum (F6(0,0)=1)')
    plt.title('Evolution of Fitness per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

# Execute o algoritmo
if __name__ == "__main__":
    genetic_algorithm_gif_plot()