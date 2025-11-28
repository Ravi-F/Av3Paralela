import os
import sys
import signal
import logging
import socket
import multiprocessing
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MatrixServer')

# Limita as threads das bibliotecas numéricas para não competir com nosso multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Constantes
MAX_WORKERS = 4  # Número máximo de workers no pool de threads
SOCKET_TIMEOUT = 30  # 30 segundos de timeout
MAX_RETRIES = 3  # Número máximo de tentativas de processamento

def receive_all(sock, length):
    """
    Recebe todos os dados de um socket até atingir o comprimento especificado.
    
    Args:
        sock: Socket de onde os dados serão recebidos
        length: Número de bytes esperados
        
    Returns:
        bytes: Dados recebidos ou None em caso de erro
    """
    try:
        data = bytearray()
        while len(data) < length:
            remaining = length - len(data)
            packet = sock.recv(min(4096, remaining))
            if not packet:
                logger.error("Conexão fechada pelo cliente durante o recebimento de dados")
                return None
            data.extend(packet)
        return bytes(data)
    except socket.timeout:
        logger.error("Timeout ao receber dados do cliente")
        return None
    except socket.error as e:
        logger.error(f"Erro de socket ao receber dados: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao receber dados: {e}")
        return None

def handle_client_connection(conn, addr, matrix_B):
    """
    Lida com uma conexão de cliente, processando a requisição recebida.
    
    Args:
        conn: Socket da conexão com o cliente
        addr: Tupla (host, port) do cliente
        matrix_B: Matriz B para uso na multiplicação (pode ser None se ainda não configurada)
    """
    try:
        conn.settimeout(SOCKET_TIMEOUT)
        logger.debug(f"Processando conexão de {addr[0]}:{addr[1]}")
        
        # Recebe o tipo de operação (SETUP ou COMPUTE)
        operation = conn.recv(8).strip()
        if not operation:
            logger.warning("Conexão fechada pelo cliente durante o handshake")
            return
            
        # Recebe o tamanho dos dados
        length_bytes = conn.recv(8)
        if not length_bytes:
            logger.error("Conexão fechada inesperadamente durante o recebimento do tamanho")
            return
            
        data_length = int.from_bytes(length_bytes, 'big')
        if data_length <= 0 or data_length > 500 * 1024 * 1024:  # Limite de 500MB
            logger.error(f"Tamanho de dados inválido: {data_length} bytes")
            conn.sendall(b'ERROR')
            return
            
        # Recebe os dados
        received_data = receive_all(conn, data_length)
        if received_data is None:
            logger.error("Falha ao receber dados do cliente")
            conn.sendall(b'ERROR')
            return
            
        # Processa a operação
        if operation == b'SETUP':
            # Fase de setup - armazena a matriz B
            try:
                new_matrix_B = pickle.loads(received_data)
                # Atualiza a referência da matriz B na thread atual
                # (em um ambiente real, você pode querer usar uma estrutura thread-safe aqui)
                global matrix_B_global
                matrix_B_global = new_matrix_B
                logger.info(f"Matriz B {new_matrix_B.shape} recebida e armazenada")
                conn.sendall(b'OK')
            except Exception as e:
                logger.error(f"Falha ao carregar a matriz B: {e}")
                conn.sendall(b'ERROR')
        elif operation == b'COMPUTE':
            # Fase de computação - processa a submatriz A
            if matrix_B_global is None:
                logger.error("Matriz B não configurada")
                conn.sendall(b'MATRIX_B_NOT_SET')
                return
                
            try:
                submatrix_A = pickle.loads(received_data)
                logger.debug(f"Processando submatriz {submatrix_A.shape} x {matrix_B_global.shape}")
                
                # Realiza a multiplicação
                result = np.dot(submatrix_A, matrix_B_global)
                
                # Envia o resultado
                serialized_result = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                conn.sendall(len(serialized_result).to_bytes(8, 'big'))
                conn.sendall(serialized_result)
                
                logger.debug(f"Resultado {result.shape} enviado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro ao processar submatriz: {e}", exc_info=True)
                conn.sendall(b'ERROR')
        else:
            logger.warning(f"Operação desconhecida: {operation}")
            conn.sendall(b'UNKNOWN_OPERATION')
            
    except (socket.timeout, socket.error) as e:
        logger.error(f"Erro de comunicação com o cliente {addr}: {e}")
    except Exception as e:
        logger.error(f"Erro inesperado ao processar cliente {addr}: {e}", exc_info=True)
    finally:
        try:
            conn.close()
        except:
            pass

# Variável global para armazenar a matriz B (simplificação para o exemplo)
matrix_B_global = None

def iniciar_servidor(host, port):
    """
    Inicia o servidor na porta especificada.
    
    Args:
        host: Endereço para escutar (geralmente '0.0.0.0' ou 'localhost')
        port: Porta para escutar conexões
    """
    # Configura o manipulador de sinal para desligamento gracioso
    def handler(signum, frame):
        logger.info(f"Recebido sinal {signum}, encerrando servidor na porta {port}...")
        server_socket.close()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
    except socket.error as e:
        logger.error(f"Erro ao vincular ao endereço {host}:{port}: {e}")
        sys.exit(1)
        
    server_socket.listen(10)
    logger.info(f"Servidor iniciado em {host}:{port}")
    
    # Matriz B compartilhada entre as threads
    matrix_B = None
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        logger.info("Aguardando conexões...")
        
        while True:
            try:
                conn, addr = server_socket.accept()
                logger.info(f"Nova conexão de {addr[0]}:{addr[1]}")
                
                # Submete a tarefa para o pool de threads
                executor.submit(handle_client_connection, conn, addr, matrix_B)
                
            except socket.timeout:
                logger.warning("Timeout ao aguardar conexões")
                continue
            except socket.error as e:
                logger.error(f"Erro de socket: {e}")
                break
            except Exception as e:
                logger.error(f"Erro inesperado: {e}", exc_info=True)
                break

def main():
   
    # Configura o nível de log baseado em uma variável de ambiente ou usa INFO como padrão
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORTS = [int(p) for p in os.environ.get('PORTS', '12345,12346').split(',')]
    
    logger.info(f"Iniciando {len(PORTS)} servidores em {HOST}:{PORTS}")
    
    processes = []
    
    def shutdown_servers(signum, frame):
        logger.info("Encerrando servidores...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_servers)
    signal.signal(signal.SIGTERM, shutdown_servers)
    
    try:
        # Inicia cada servidor em seu próprio processo
        for port in PORTS:
            try:
                p = multiprocessing.Process(
                    target=iniciar_servidor,
                    args=(HOST, port),
                    daemon=True
                )
                p.start()
                processes.append(p)
                logger.info(f"Servidor iniciado na porta {port} (PID: {p.pid})")
            except Exception as e:
                logger.error(f"Falha ao iniciar servidor na porta {port}: {e}")
        
        # Mantém o processo principal vivo
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        logger.info("Interrupção recebida, encerrando...")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}", exc_info=True)
    finally:
        shutdown_servers(None, None)

if __name__ == "__main__":
    main()
