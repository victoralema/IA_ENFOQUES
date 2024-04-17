#Victor Eduardo Aleman Padilla 21310193


import numpy as np

class GraphQLearning:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.Q = np.zeros((num_nodes, num_nodes))  # Matriz Q inicializada con ceros
        self.learning_rate = 0.8
        self.discount_factor = 0.9

    def choose_action(self, state, possible_actions, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            # Exploración: elegir una acción aleatoria entre las posibles
            return np.random.choice(possible_actions)
        else:
            # Explotación: elegir la acción con el valor Q máximo para el estado actual
            Q_state = self.Q[state, possible_actions]
            return possible_actions[np.argmax(Q_state)]

    def update_q_value(self, state, next_state, action, reward):
        best_next_action = np.argmax(self.Q[next_state])
        self.Q[state, action] += self.learning_rate * (reward + self.discount_factor * self.Q[next_state, best_next_action] - self.Q[state, action])

    def find_shortest_path(self, graph, start_node, goal_node, num_episodes=1000, epsilon=0.2):
        for episode in range(num_episodes):
            current_state = start_node
            path = [current_state]
            
            while current_state != goal_node:
                possible_actions = np.where(graph[current_state] >= 0)[0]  # Obtener acciones válidas desde el estado actual
                action = self.choose_action(current_state, possible_actions, epsilon)
                next_state = action
                reward = graph[current_state, next_state]
                
                # Actualizar el valor Q del estado actual
                self.update_q_value(current_state, next_state, action, reward)
                
                # Mover al siguiente estado
                current_state = next_state
                path.append(current_state)
            
            if path[-1] == goal_node:
                print(f"Camino más corto encontrado en el episodio {episode}: {path}")

        return self.Q

# Ejemplo de uso
if __name__ == "__main__":
    # Definir el grafo como una matriz de adyacencia ponderada
    # Aquí, -1 indica que no hay conexión entre los nodos
    graph = np.array([
        [-1, 2, -1, 1, -1, -1],  # Nodo 0
        [-1, -1, 3, -1, -1, -1], # Nodo 1
        [-1, -1, -1, -1, 4, -1],  # Nodo 2
        [-1, -1, -1, -1, 3, 7],   # Nodo 3
        [-1, -1, -1, -1, -1, 1],  # Nodo 4
        [-1, -1, -1, -1, -1, -1]  # Nodo 5 (Objetivo)
    ])

    num_nodes = len(graph)
    start_node = 0
    goal_node = 5

    ql = GraphQLearning(num_nodes)
    learned_Q = ql.find_shortest_path(graph, start_node, goal_node)

    print("\nMatriz Q final:")
    print(learned_Q)
