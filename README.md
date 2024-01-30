# Genetic Algorithm with Snake 🐍
Using a neural net as a genetic representation of the game controller and measuring fitness via a combination of score (the length of the snake) and number of steps before it crashed into a wall or did an Ouroboros🔄.

The Snake game implementation uses pygame for the engine and using a linked-list to store the body locations.

# Neural Net

Input Layer: Gamestate - grid containing body and food locations + the direction the snake moved in.

# Elitism
The top 10% of each population are selected for the next round. We duplicate these 'elite' NNs and add mutations to get the randomness needed for directed evoution. This guarantees fitness will never regress.

# Mutations
Point mutations
- Drop out
- Reroll the weight
- Sign change (multiply by -1)
Parents
- cross-over/recombination w/ speciation heuristic (penalizes crossover between too similar NNs to encourage population diversity)
- genetic swap





# Run on Linux

    mkdir pygame_snake
    cd pygame_snake
    
    python -m venv .venv
    source .venv/bin/activate

    pip install pygame

    python app.py