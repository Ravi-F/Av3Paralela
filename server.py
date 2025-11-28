import socket
import multiprocessing
import numpy as np
import pickle

def matriz_multiplicacao(submatriz_A, matriz_B):
    return np.dot(submatriz_A, matriz_B)

def tratar_cliente(client_socket):
    try:
        # Receber o tamanho da data
        tamanho_bytes = client_socket.recv(4)
        if not tamanho_bytes:
            return  # Conexão fechada

        tamanho_mensagem = int.from_bytes(tamanho_bytes, 'big')

        # Receber data
        receber_data = b''
        while len(receber_data) < tamanho_mensagem:
            pacote = client_socket.recv(min(4096, tamanho_mensagem - len(receber_data)))
            if not pacote:
                break
            receber_data += pacote

        if not receber_data:
            return  # Conexão fechada

        data = pickle.loads(receber_data)
        submatriz_A = data['A']
        matriz_B = data['B']

        # Perfoma a multiplicação de matrizes em paralelo
        resultado = matriz_multiplicacao(submatriz_A, matriz_B)

        # Serializar e enviar o resultado de volta ao cliente
        resultado_serializado = pickle.dumps(resultado)
        resultado_tamanho = len(resultado_serializado)
        client_socket.sendall(resultado_tamanho.to_bytes(4, 'big'))
        client_socket.sendall(resultado_serializado)

    except Exception as e:
        print(f"Erro no cliente: {e}")
    finally:
        client_socket.close()

def iniciar_servidor(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Servidor iniciado em {host}:{port}")

    while True:
        client_socket, address = server_socket.accept()
        print(f"Conexão estabelecida com {address}")
        tratar_cliente(client_socket)

if __name__ == "__main__":
    # Definir host e portas
    HOST = 'localhost'
    PORTS = [12345, 12346]

    # Criar processos para cada servidor
    for port in PORTS:
        process = multiprocessing.Process(target = iniciar_servidor, args=(HOST, port))
        process.start()
