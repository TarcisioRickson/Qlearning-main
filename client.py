import connection as cn
import random
import numpy as np

s = cn.connect(2037)


# Função para mapear número para ação
def conversionChoice(num):
    print(num)
    if num == 0:
        return "left"
    elif num == 1:
        return "right"
    elif num == 2:
        return "jump"

# Função para converter plataforma e direção de binários para decimal
def conversion(state):
    platform = str(state)[2:7]
    direction = str(state)[7:9]

    platform_inversed = ''.join(reversed(platform))
    direction_inversed = ''.join(reversed(direction))

    iterations = 0
    platform_dec = 0
    direction_dec = 0

    for i in platform_inversed:
        if i == '1':
            platform_dec += 2 ** iterations

        iterations += 1

    iterations = 0

    for j in direction_inversed:
        if j == '1':
            direction_dec += 2 ** iterations

        iterations += 1

    return (platform_dec, direction_dec)


# Função para mapear plataforma/direção para estado
def get_state(platform, direction):
    state = (platform * 4) + direction
    return state


# Função para retornar a melhor ação para cada estado
def best_action(state_index, q_table):
    if (q_table[state_index, 0] > q_table[state_index, 1]) and (q_table[state_index, 0] > q_table[state_index, 2]):
        action_index = 0
    elif (q_table[state_index, 1] > q_table[state_index, 0]) and (q_table[state_index, 1] > q_table[state_index, 2]):
        action_index = 1
    else:
        action_index = 2

    return action_index


# Função de atualização da Q-table

def q_update(q_table, state, action, next_state, rw, alpha, gamma):
    estimate_q = rw + gamma * max(q_table[next_state, 0], q_table[next_state, 1], q_table[next_state, 2])

    q_value = q_table[state, action] + alpha * (estimate_q - q_table[state, action])

    return q_value


# Carregar Q-table de arquivo
q_table = np.loadtxt('resultado.txt')

# Configurar precisão de exibição
np.set_printoptions(precision=6)

# Estado inicial
state = (0, 0)

alpha = 0.1  # taxa de aprendizagem que diz o quão rápido o agente aprende
gamma = 1  # fator de desconto, diz o peso da recompensa futura em relação à imediata
epsilon = 0  # epsilon greedy strategy -> uma taxa que define se o agente irá tomar ações aleatórias ou embasadas

while (True):
    # Gerar ação aleatória
    random_num = random.randint(0, 2)
    random_action = conversionChoice(random_num)

    # Obter índice do estado atual
    state_index = get_state(state[0], state[1])

    # Gerar melhor ação para o estado atual
    based_num = best_action(state_index, q_table)
    based_action = conversionChoice(based_num)

    # Gerar número aleatório entre 0 e 1
    random_float = random.uniform(0, 1)

    # caso o valor gerado seja maior que epsilon o agente irá usar a q_table para tomar a ação, caso contrário será aleatória sua tomada de decisão

    if (random_float >= epsilon):
        action_num = based_num
        action = based_action
    else:  # epsilon = 0 -> ações sempre usando a q_table
        action_num = random_num  # epsilon = 1 -> ações sempre aleatórias
        action = random_action

    # Executar ação e obter próximo estado e recompensa
    next_state, rw = cn.get_state_reward(s, action)

    # Exibir informações para monitoramento
    print(f'action:{action}')
    print(f'state:{next_state}')
    print(f'bounty:{rw}')

    # Converter próximo estado para formato decimal
    next_state = conversion(next_state)
    next_state_index = get_state(next_state[0], next_state[1])

    # Atualizar Q-table
    q_table[state_index, action_num] = q_update(q_table, state_index, action_num, next_state_index, rw, alpha, gamma)

    # escreve no resultado.txt
    np.savetxt('resultado.txt', q_table, fmt="%f")

    # Atualizar estado atual
    state = next_state
