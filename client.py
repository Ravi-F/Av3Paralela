import os
#Para limitar o uso de threads em bibliotecas numéricas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import socket
import multiprocessing as mp
import pickle
import time

def split_matrix(matrix, num_parts):
    return np.array_split(matrix, num_parts, axis=0)

def gerar_matriz(rows, cols):
    print(f"Gerando matriz de tamanho {rows}x{cols}")
    return np.random.randint(0, 10, size=(rows, cols))

def connect_and_send(server_address, submatrix_A, matrix_B):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(server_address)

        # Prepara os dados para envio
        data = {'A': submatrix_A, 'B': matrix_B}
        serialized_data = pickle.dumps(data)

        # Envia os dados com um cabeçalho indicando o tamanho
        length = len(serialized_data)
        client_socket.sendall(length.to_bytes(4, 'big'))  # Envia o tamanho como 4 bytes
        client_socket.sendall(serialized_data)

        # Recebe o resultado
        result_length_bytes = client_socket.recv(4)
        result_length = int.from_bytes(result_length_bytes, 'big')
        received_data = b''
        while len(received_data) < result_length:
            pacote = client_socket.recv(min(4096, result_length - len(received_data)))
            if not pacote:
                break
            received_data += pacote
        partial_result = pickle.loads(received_data)

        client_socket.close()
        return partial_result

    except Exception as e:
        print(f"Erro no processo de trabalho: {e}")
        return None

def multiplicacao_serial(A, B):
    """Executa a multiplicação de forma sequencial (serial, 1 thread)."""
    print("\nIniciando multiplicação serial (1 thread)...")
    start_time = time.time()
    result = np.dot(A, B)
    end_time = time.time()
    print(f"-> Multiplicação serial concluída em: {end_time - start_time:.4f} segundos")
    return result

def multiplicacao_paralela(A, B):
    """Executa a multiplicação de forma paralela usando múltiplos processos."""
    print("\nIniciando multiplicação paralela...")
    start_time = time.time()

    num_processes = mp.cpu_count()
    submatrices_A = split_matrix(A, num_processes)

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(np.dot, [(sub_A, B) for sub_A in submatrices_A])

    final_result = np.concatenate(results, axis=0)

    end_time = time.time()
    print(f"-> Multiplicação paralela concluída em: {end_time - start_time:.4f} segundos")
    return final_result

def multiplicacao_distribuida(A, B):
    """Executa a multiplicação de forma distribuída (cliente-servidor)."""
    print("\nIniciando multiplicação distribuída...")
    start_time = time.time()
    server_addresses = [('localhost', 12345), ('localhost', 12346)]
    num_servers = len(server_addresses)
    submatrices_A = split_matrix(A, num_servers)

    tasks = [(server_addresses[i], submatrices_A[i], B) for i in range(num_servers)]

    with mp.Pool(processes=num_servers) as pool:
        results = pool.starmap(connect_and_send, tasks)

    if any(res is None for res in results):
        print("-> Erro: Falha na comunicação com um ou mais servidores.")
        return None

    final_result = np.concatenate(results, axis=0)
    end_time = time.time()
    print(f"-> Multiplicação distribuída concluída em: {end_time - start_time:.4f} segundos")
    return final_result


def main():
    #A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    # Lista de casos de teste com diferentes dimensões de matrizes
    # Formato: (linhas_A, colunas_A_e_linhas_B, colunas_B)
    casos_de_teste = [
        (100, 100, 100),   # Pequena quadrada
        (200, 200, 200),
        (500, 500, 500),   # Média quadrada (original)
        (800, 800, 800),
        (1000, 1000, 1000),  # Grande quadrada
        (1200, 1200, 1200),
        (1500, 1500, 1500),  # Muito grande quadrada
        (100, 500, 200),   # Retangulares
        (500, 100, 200),
        (1000, 200, 800),
        (200, 1000, 800),
        (1500, 100, 1500),  # A alta e fina
        (100, 1500, 100),  # A baixa e larga
        (20, 2000, 20),
        (2000, 20, 2000),
        (600, 700, 800)    # Dimensões variadas
    ]

    for i, (rows_a, cols_a_rows_b, cols_b) in enumerate(casos_de_teste):
        print(f"\n--- CASO DE TESTE {i+1}/{len(casos_de_teste)}: A({rows_a}x{cols_a_rows_b}), B({cols_a_rows_b}x{cols_b}) ---")
        A = gerar_matriz(rows_a, cols_a_rows_b)
        B = gerar_matriz(cols_a_rows_b, cols_b)
        _ = multiplicacao_serial(A, B)
        _ = multiplicacao_paralela(A, B)
        _ = multiplicacao_distribuida(A, B)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTempo total de execução: {end_time - start_time:.4f} segundos")
