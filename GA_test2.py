import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Fixed GA Parameters (Exam Requirement) ----
POP_SIZE = 300
CHROM_LEN = 80
FITNESS_PEAKS_AT_ONES = 40
MAX_FITNESS = 80
N_GENERATIONS = 50

# ---- GA Hyperparameters ----
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN

# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - FITNESS_PEAKS_AT_ONES)

# ---- GA Operators ----
def init_population(pop_size, chrom_len):
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

def tournament_selection(pop, fits, k):
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def single_point_crossover(p1, p2):
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(ind):
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    ind[mask] = 1 - ind[mask]
    return ind

def evolve(pop):
    best_curve = []
    best_individual = None
    best_f = -np.inf

    for _ in range(N_GENERATIONS):
        fits = np.array([fitness(ind) for ind in pop])

        gen_best_idx = np.argmax(fits)
        gen_best_f = fits[gen_best_idx]
        best_curve.append(float(gen_best_f))

        if gen_best_f > best_f:
            best_f = float(gen_best_f)
            best_individual = pop[gen_best_idx].copy()

        new_pop = []
        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:len(pop)], dtype=np.int8)

    return best_individual, best_f, best_curve

# ---- Streamlit UI ----
st.set_page_config(page_title="GA Bit Pattern")

st.title("Genetic Algorithm – Bit Pattern Generator")

st.write("""
**Fixed Parameters (as required):**
- Population = 300  
- Chromosome length = 80  
- Generations = 50  
- Fitness peaks at ones = 40  
- Max fitness = 80  
""")

seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

if st.button("Run Genetic Algorithm", type="primary"):
    random.seed(seed)
    np.random.seed(seed)

    pop = init_population(POP_SIZE, CHROM_LEN)
    best_ind, best_fit, curve = evolve(pop)

    ones = int(best_ind.sum())
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("Best Individual Found")
    st.metric("Best Fitness", f"{best_fit:.0f}")
    st.write(f"Ones = {ones}, Zeros = {CHROM_LEN - ones}")
    st.code(bitstring)

    st.subheader("Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(curve)+1), curve)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("GA Convergence Curve")
    ax.grid(True)
    st.pyplot(fig)

    if best_fit == MAX_FITNESS and ones == FITNESS_PEAKS_AT_ONES:
        st.success("Optimal solution reached (40 ones, fitness = 80) ✅")
    else:
        st.info("Near-optimal solution found. Try different seed for variation.")

