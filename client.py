import os
import sys
import time
import socket
import pickle
import signal
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# Cria diretórios se não existirem
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
HTML_REPORTS_DIR = BASE_DIR / "html_reports"

LOG_DIR.mkdir(exist_ok=True)
HTML_REPORTS_DIR.mkdir(exist_ok=True)

# Configuração de logging
log_file = LOG_DIR / f"matrix_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger('MatrixClient')
logger.info(f"Logs serão salvos em: {log_file}")
logger.info(f"Relatórios HTML serão salvos em: {HTML_REPORTS_DIR}")

# Limita as threads das bibliotecas numéricas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Constantes
SOCKET_TIMEOUT = 30  # segundos
MAX_RETRIES = 3      # Número máximo de tentativas
CHUNK_SIZE = 4096    # Tamanho do chunk para envio/recebimento

def split_matrix(matrix, num_parts):
    """
    Divide uma matriz em submatrizes ao longo do eixo 0.
    
    Args:
        matrix: Matriz numpy a ser dividida
        num_parts: Número de partes para dividir
        
    Returns:
        Lista de submatrizes
    """
    try:
        if not isinstance(matrix, np.ndarray):
            raise ValueError("A entrada deve ser uma matriz numpy")
        if num_parts <= 0:
            raise ValueError("O número de partes deve ser maior que zero")
            
        return np.array_split(matrix, num_parts, axis=0)
        
    except Exception as e:
        logger.error(f"Erro ao dividir matriz: {e}")
        raise

def gerar_matriz(rows, cols, min_val=0, max_val=10):
    """
    Gera uma matriz aleatória com valores inteiros.
    
    Args:
        rows: Número de linhas
        cols: Número de colunas
        min_val: Valor mínimo (inclusivo)
        max_val: Valor máximo (exclusivo)
        
    Returns:
        Matriz numpy de tamanho rows x cols
    """
    try:
        logger.info(f"Gerando matriz de tamanho {rows}x{cols}")
        if rows <= 0 or cols <= 0:
            raise ValueError("Dimensões da matriz devem ser positivas")
            
        return np.random.randint(min_val, max_val, size=(rows, cols), dtype=np.int32)
        
    except Exception as e:
        logger.error(f"Erro ao gerar matriz: {e}")
        raise

def setup_server(server_address, matrix_B, timeout=SOCKET_TIMEOUT):
    """
    FASE 1: Envia a matriz B para um servidor e espera confirmação.
    
    Args:
        server_address: Tupla (host, port) do servidor
        matrix_B: Matriz B a ser enviada
        timeout: Timeout em segundos
        
    Returns:
        bool: True se o setup foi bem-sucedido, False caso contrário
    """
    client_socket = None
    
    for attempt in range(MAX_RETRIES):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(timeout)
            
            logger.debug(f"Conectando ao servidor {server_address} (tentativa {attempt + 1}/{MAX_RETRIES})")
            
            try:
                client_socket.connect(server_address)
            except socket.error as e:
                logger.error(f"Não foi possível conectar ao servidor {server_address}: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(1)  # Espera antes de tentar novamente
                continue
            
            try:
                # Envia o tipo de operação (8 bytes)
                client_socket.sendall(b'SETUP   ')  # 8 bytes
                
                # Serializa a matriz B
                serialized_data = pickle.dumps(matrix_B, protocol=pickle.HIGHEST_PROTOCOL)
                data_length = len(serialized_data)
                
                # Envia o tamanho dos dados
                client_socket.sendall(data_length.to_bytes(8, 'big'))
                
                # Envia os dados em chunks
                total_sent = 0
                while total_sent < data_length:
                    sent = client_socket.send(serialized_data[total_sent:total_sent + CHUNK_SIZE])
                    if sent == 0:
                        raise RuntimeError("Conexão fechada pelo servidor durante o envio")
                    total_sent += sent
                
                # Aguarda confirmação
                response = client_socket.recv(2)  # 'OK' ou 'ER'
                
                if response == b'OK':
                    logger.info(f"Setup concluído com sucesso no servidor {server_address}")
                    return True
                else:
                    logger.warning(f"Resposta inesperada do servidor {server_address}: {response}")
                    
            except (socket.timeout, socket.error) as e:
                logger.warning(f"Erro de comunicação com o servidor {server_address}: {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Falha após {MAX_RETRIES} tentativas no servidor {server_address}")
            except Exception as e:
                logger.error(f"Erro inesperado durante o setup no servidor {server_address}: {e}", exc_info=True)
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Falha após {MAX_RETRIES} tentativas no servidor {server_address}")
            
        except Exception as e:
            logger.error(f"Erro inesperado: {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return False
        finally:
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass
    
    return False

def compute_on_server(server_address, submatrix_A, timeout=SOCKET_TIMEOUT):
    """
    FASE 2: Envia uma parte de A para um servidor e recebe o resultado.
    
    Args:
        server_address: Tupla (host, port) do servidor
        submatrix_A: Submatriz de A para processamento
        timeout: Timeout em segundos
        
    Returns:
        Matriz de resultado ou None em caso de falha
    """
    client_socket = None
    
    for attempt in range(MAX_RETRIES):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(timeout)
            
            logger.debug(f"Conectando ao servidor {server_address} para processamento (tentativa {attempt + 1}/{MAX_RETRIES})")
            
            try:
                client_socket.connect(server_address)
            except socket.error as e:
                logger.error(f"Não foi possível conectar ao servidor {server_address}: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(1)  # Espera antes de tentar novamente
                continue
            
            try:
                # Envia o tipo de operação (8 bytes)
                client_socket.sendall(b'COMPUTE ')  # 8 bytes
                
                # Serializa a submatriz A
                serialized_data = pickle.dumps(submatrix_A, protocol=pickle.HIGHEST_PROTOCOL)
                data_length = len(serialized_data)
                
                # Envia o tamanho dos dados
                client_socket.sendall(data_length.to_bytes(8, 'big'))
                
                # Envia os dados em chunks
                total_sent = 0
                while total_sent < data_length:
                    sent = client_socket.send(serialized_data[total_sent:total_sent + CHUNK_SIZE])
                    if sent == 0:
                        raise RuntimeError("Conexão fechada pelo servidor durante o envio")
                    total_sent += sent
                
                # Recebe o tamanho da resposta
                result_length_bytes = client_socket.recv(8)
                if not result_length_bytes:
                    raise RuntimeError("Conexão fechada pelo servidor durante o recebimento do tamanho")
                    
                result_length = int.from_bytes(result_length_bytes, 'big')
                
                if result_length <= 0 or result_length > 1024 * 1024 * 1024:  # Limite de 1GB
                    raise ValueError(f"Tamanho de resposta inválido do servidor: {result_length} bytes")
                
                # Recebe os dados em chunks
                received_data = bytearray()
                while len(received_data) < result_length:
                    chunk = client_socket.recv(min(CHUNK_SIZE, result_length - len(received_data)))
                    if not chunk:
                        raise RuntimeError("Conexão fechada pelo servidor durante o recebimento dos dados")
                    received_data.extend(chunk)
                
                # Desserializa o resultado
                result = pickle.loads(received_data)
                
                if not isinstance(result, np.ndarray):
                    raise ValueError(f"Resposta inválida do servidor: esperado ndarray, obtido {type(result)}")
                
                logger.debug(f"Resultado recebido do servidor {server_address}: {result.shape}")
                return result
                
            except (socket.timeout, socket.error) as e:
                logger.warning(f"Erro de comunicação com o servidor {server_address}: {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Falha após {MAX_RETRIES} tentativas no servidor {server_address}")
                    return None
                time.sleep(1)  # Espera antes de tentar novamente
                
            except Exception as e:
                logger.error(f"Erro inesperado durante o processamento no servidor {server_address}: {e}", exc_info=True)
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(1)  # Espera antes de tentar novamente
                
        except Exception as e:
            logger.error(f"Erro inesperado: {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(1)  # Espera antes de tentar novamente
            
        finally:
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass
    
    return None

def multiplicacao_serial(A, B):
    """
    Executa a multiplicação de matrizes de forma sequencial.
    
    Args:
        A: Primeira matriz (m x n)
        B: Segunda matriz (n x p)
        
    Returns:
        Tupla (matriz_resultado, tempo_execucao)
    """
    logger.info("Iniciando multiplicação serial (1 thread)...")
    
    try:
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Dimensões incompatíveis para multiplicação: {A.shape} e {B.shape}")
            
        start_time = time.perf_counter()
        result = np.dot(A, B)
        exec_time = time.perf_counter() - start_time
        
        logger.info(f"Multiplicação serial concluída em: {exec_time:.4f} segundos")
        return result, exec_time
        
    except Exception as e:
        logger.error(f"Erro na multiplicação serial: {e}", exc_info=True)
        raise

def multiplicacao_paralela(A, B, num_processes=None):
    """
    Executa a multiplicação de matrizes de forma paralela usando multiprocessing.
    
    Args:
        A: Primeira matriz (m x n)
        B: Segunda matriz (n x p)
        num_processes: Número de processos a serem usados (padrão: número de CPUs)
        
    Returns:
        Tupla (matriz_resultado, tempo_execucao)
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Limita a 8 processos no máximo
        
    logger.info(f"Iniciando multiplicação paralela com {num_processes} processos...")
    
    try:
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Dimensões incompatíveis para multiplicação: {A.shape} e {B.shape}")
            
        start_time = time.perf_counter()
        
        # Divide a matriz A em submatrizes
        submatrices_A = split_matrix(A, num_processes)
        
        # Cria um pool de processos
        with mp.Pool(processes=num_processes) as pool:
            # Mapeia cada submatriz para um processo
            results = pool.starmap(np.dot, [(sub_A, B) for sub_A in submatrices_A])
        
        # Combina os resultados
        final_result = np.concatenate(results, axis=0)
        
        exec_time = time.perf_counter() - start_time
        logger.info(f"Multiplicação paralela concluída em: {exec_time:.4f} segundos")
        return final_result, exec_time
        
    except Exception as e:
        logger.error(f"Erro na multiplicação paralela: {e}", exc_info=True)
        raise

def multiplicacao_distribuida(A, B, server_addresses=None):
    """
    Executa a multiplicação de matrizes de forma distribuída usando múltiplos servidores.
    
    Args:
        A: Primeira matriz (m x n)
        B: Segunda matriz (n x p)
        server_addresses: Lista de tuplas (host, port) dos servidores
        
    Returns:
        Tupla (matriz_resultado, tempo_execucao)
    """
    if server_addresses is None:
        server_addresses = [('localhost', 12345), ('localhost', 12346)]
        
    num_servers = len(server_addresses)
    if num_servers == 0:
        raise ValueError("Nenhum servidor especificado")
        
    logger.info(f"Iniciando multiplicação distribuída com {num_servers} servidores...")
    logger.info(f"Dimensões: A={A.shape}, B={B.shape}")
    
    try:
        # Verifica se as dimensões são compatíveis para multiplicação
        if A.shape[1] != B.shape[0]:
            error_msg = f"Dimensões incompatíveis para multiplicação: A {A.shape} e B {B.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        start_time = time.perf_counter()
        
        # --- FASE 1: SETUP ---
        # Envia a matriz B para todos os servidores
        logger.info(f"Enviando matriz B ({B.shape}) para {num_servers} servidores...")
        with ThreadPoolExecutor(max_workers=num_servers) as executor:
            setup_tasks = {executor.submit(setup_server, addr, B.copy()): addr for addr in server_addresses}
            
            # Aguarda a conclusão de todas as tarefas de setup
            setup_success = True
            for future in as_completed(setup_tasks):
                addr = setup_tasks[future]
                try:
                    success = future.result()
                    if not success:
                        logger.error(f"Falha no setup do servidor {addr}")
                        setup_success = False
                except Exception as e:
                    logger.error(f"Erro durante o setup do servidor {addr}: {e}")
                    setup_success = False
            
            if not setup_success:
                error_msg = "Falha na fase de SETUP com um ou mais servidores. Verifique os logs para mais detalhes."
                logger.error(error_msg)
                logger.error("!!! REINICIE OS SERVIDORES ANTES DA PRÓXIMA EXECUÇÃO !!!")
                return None, 0
        
        # --- FASE 2: COMPUTAÇÃO ---
        # Divide a matriz A e envia para os servidores
        logger.info(f"Dividindo matriz A ({A.shape}) em {num_servers} partes...")
        submatrices_A = split_matrix(A, num_servers)
        
        # Verifica se o número de submatrizes é igual ao número de servidores
        if len(submatrices_A) != num_servers:
            error_msg = f"Número de submatrizes ({len(submatrices_A)}) não corresponde ao número de servidores ({num_servers})"
            logger.error(error_msg)
            return None, 0
            
        logger.info(f"Enviando submatrizes para processamento...")
        
        with ThreadPoolExecutor(max_workers=num_servers) as executor:
            # Envia cada submatriz para um servidor diferente
            compute_tasks = {}
            for i, sub_A in enumerate(submatrices_A):
                if sub_A.size > 0:  # Apenas envia submatrizes não vazias
                    compute_tasks[executor.submit(compute_on_server, server_addresses[i], sub_A.copy())] = i
                else:
                    logger.warning(f"Submatriz {i} está vazia, pulando...")
            
            if not compute_tasks:
                error_msg = "Nenhuma submatriz válida para processamento"
                logger.error(error_msg)
                return None, 0
            
            # Coleta os resultados
            results = [None] * num_servers
            for future in as_completed(compute_tasks):
                task_idx = compute_tasks[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[task_idx] = result
                        logger.debug(f"Resultado recebido do servidor {server_addresses[task_idx]}: {result.shape}")
                    else:
                        error_msg = f"Falha ao processar submatriz no servidor {server_addresses[task_idx]}"
                        logger.error(error_msg)
                        return None, 0
                except Exception as e:
                    error_msg = f"Erro durante o processamento no servidor {server_addresses[task_idx]}: {e}"
                    logger.error(error_msg, exc_info=True)
                    return None, 0
        
        # Filtra resultados None (caso alguma submatriz tenha sido pulada)
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            error_msg = "Nenhum resultado válido retornado pelos servidores"
            logger.error(error_msg)
            return None, 0
            
        # Combina os resultados
        try:
            logger.info("Combinando resultados parciais...")
            final_result = np.concatenate(valid_results, axis=0)
            
            # Verifica se as dimensões do resultado estão corretas
            expected_rows = A.shape[0]
            expected_cols = B.shape[1]
            if final_result.shape != (expected_rows, expected_cols):
                error_msg = (
                    f"Dimensões incorretas do resultado. "
                    f"Esperado: ({expected_rows}, {expected_cols}), "
                    f"Obtido: {final_result.shape}"
                )
                logger.error(error_msg)
                return None, 0
                
            exec_time = time.perf_counter() - start_time
            logger.info(f"Multiplicação distribuída concluída em: {exec_time:.4f} segundos")
            logger.info(f"Resultado final: {final_result.shape}")
            
            return final_result, exec_time
            
        except Exception as e:
            error_msg = f"Erro ao combinar resultados: {e}"
            logger.error(error_msg, exc_info=True)
            return None, 0
        
    except Exception as e:
        logger.error(f"Erro na multiplicação distribuída: {e}", exc_info=True)
        return None, 0


def main():
    """
    Função principal que executa os testes de desempenho para diferentes tamanhos de matrizes.
    Compara os tempos de execução entre multiplicação serial, paralela e distribuída.
    """
    try:
        # Nomes dos arquivos de saída
        csv_filename = "resultados_comparativos.csv"
        plot_filename = "comparativo_execucao.png"
        
        
        logger.info("Iniciando execução do cliente de multiplicação de matrizes")
        
        # Remove os arquivos de resultados anteriores para garantir que sejam sempre novos
        if os.path.exists(csv_filename):
            try:
                os.remove(csv_filename)
                logger.info(f"Arquivo de resultado anterior '{csv_filename}' removido.")
            except Exception as e:
                logger.warning(f"Não foi possível remover o arquivo {csv_filename}: {e}")
                
        if os.path.exists(plot_filename):
            try:
                os.remove(plot_filename)
                logger.info(f"Gráfico anterior '{plot_filename}' removido.")
            except Exception as e:
                logger.warning(f"Não foi possível remover o arquivo {plot_filename}: {e}")
        
        # Limpa relatórios HTML antigos antes de começar
        import glob
        for arquivo in glob.glob(str(HTML_REPORTS_DIR / '*.html')):
            try:
                os.remove(arquivo)
                logger.info(f'Arquivo removido: {arquivo}')
            except Exception as e:
                logger.warning(f'Erro ao remover {arquivo}: {e}')
    
        # Lista de casos de teste com diferentes dimensões de matrizes
        # Formato: (linhas_A, colunas_A_e_linhas_B, colunas_B)
        casos_de_teste = [
            (2, 2, 2, 2),           # Caso 1: 2x2 * 2x2 (pequeno para demonstração)
            (200, 200, 200, 200),   # Caso 2: 200x200 * 200x200
            (300, 300, 300, 300),   # Caso 3: 300x300 * 300x300
            (400, 400, 400, 400),   # Caso 4: 400x400 * 400x400
            (500, 500, 500, 500),   # Caso 5: 500x500 * 500x500
            (100, 200, 200, 300),   # Caso 6: 100x200 * 200x300
            (200, 300, 300, 400),   # Caso 7: 200x300 * 300x400
            (300, 400, 400, 500),   # Caso 8: 300x400 * 400x500
            (400, 500, 500, 600),   # Caso 9: 400x500 * 500x600
            (500, 600, 600, 700),   # Caso 10: 500x600 * 600x700
            (600, 700, 700, 800),   # Caso 11: 600x700 * 700x800
            (700, 800, 800, 900),   # Caso 12: 700x800 * 800x900
            (800, 900, 900, 1000),  # Caso 13: 800x900 * 900x1000
            (900, 1000, 1000, 1100),# Caso 14: 900x1000 * 1000x1100
            (1000, 1000, 1000, 1000), # Caso 15: 1000x1000 * 1000x1000
            (1500, 1500, 1500, 1500)  # Caso 16: 1500x1500 * 1500x1500
        ]
        
        num_execucoes_por_caso = 3  # Reduzido para tornar os testes mais rápidos
        
        # Configuração dos servidores
        server_addresses = [('localhost', 12345), ('localhost', 12346)]
        
        logger.info(f"Configuração: {num_execucoes_por_caso} execuções por caso de teste")
        logger.info(f"Endereços dos servidores: {server_addresses}")
        
        resultados_execucao = []
        
        for i, (rows_a, cols_a_rows_b, cols_b, _) in enumerate(casos_de_teste):
            dim_label = f"A({rows_a}x{cols_a_rows_b}), B({cols_a_rows_b}x{cols_b})"
            logger.info(f"\n--- CASO DE TESTE {i+1}/{len(casos_de_teste)}: {dim_label} ---")
            
            tempos_serial, tempos_paralelo, tempos_distribuido = [], [], []
            
            for j in range(num_execucoes_por_caso):
                logger.info(f"Execução {j+1}/{num_execucoes_por_caso}...")
                
                try:
                    # Gera as matrizes para este teste
                    logger.debug(f"Gerando matrizes para o caso {i+1}, execução {j+1}")
                    A = gerar_matriz(rows_a, cols_a_rows_b)
                    B = gerar_matriz(cols_a_rows_b, cols_b)
                    
                    # Executa a multiplicação serial
                    logger.debug("Iniciando multiplicação serial...")
                    _, tempo_s = multiplicacao_serial(A, B)
                    tempos_serial.append(tempo_s)
                    
                    # Executa a multiplicação paralela
                    logger.debug("Iniciando multiplicação paralela...")
                    _, tempo_p = multiplicacao_paralela(A, B)
                    tempos_paralelo.append(tempo_p)
                    
                    # Executa a multiplicação distribuída
                    logger.debug("Iniciando multiplicação distribuída...")
                    _, tempo_d = multiplicacao_distribuida(A, B, server_addresses)
                    if tempo_d is not None:  # Pode ser None em caso de falha
                        tempos_distribuido.append(tempo_d)
                    
                    logger.info(f"Tempos - Serial: {tempo_s:.4f}s, Paralelo: {tempo_p:.4f}s, Distribuído: {tempo_d if tempo_d is not None else 'N/A'}s")

                except Exception as e:
                    logger.error(f"Erro durante a execução do caso {i+1}, tentativa {j+1}: {e}", exc_info=True)
                    # Continua para a próxima iteração, mas registra o erro
                    continue

            # Calcula as médias, garantindo que temos pelo menos uma medição válida
            media_serial = np.mean(tempos_serial) if tempos_serial else float('nan')
            media_paralelo = np.mean(tempos_paralelo) if tempos_paralelo else float('nan')
            media_distribuido = np.mean(tempos_distribuido) if tempos_distribuido else float('nan')

            # Loga os resultados para depuração
            logger.debug(f"Médias - Serial: {media_serial:.4f}s, Paralelo: {media_paralelo:.4f}s, Distribuído: {media_distribuido if not np.isnan(media_distribuido) else 'N/A'}s")
            if not np.isnan(media_serial) and not np.isnan(media_distribuido):
                speedup_dist = media_serial / media_distribuido if media_distribuido > 0 else float('nan')
                logger.debug(f"Speedup Distribuído calculado: {speedup_dist:.2f}x")

            # Adiciona os resultados à lista
            resultados_execucao.append({
                "Caso": f"Caso {i+1}",
                "Dimensoes": dim_label,
                "Operacoes": rows_a * cols_a_rows_b * cols_b,
                "Serial (s)": media_serial,
                "Paralelo (s)": media_paralelo,
                "Distribuido (s)": media_distribuido,
                "Speedup Paralelo": media_serial / media_paralelo if media_paralelo > 0 else float('nan'),
                "Speedup Distribuido": media_serial / media_distribuido if media_distribuido > 0 else float('nan')
            })
            
            # Salva resultados parciais a cada caso de teste
            df = pd.DataFrame(resultados_execucao)
            df.to_csv(csv_filename, index=False, float_format='%.4f')
            logger.info(f"Resultados parciais salvos em '{csv_filename}'")
            
            # Gera o relatório HTML com as matrizes e resultados
            if not np.isnan(media_serial) and not np.isnan(media_paralelo):
                try:
                    # Gera matrizes de exemplo para o relatório
                    # (já que não temos acesso às matrizes originais neste ponto)
                    rows_a, cols_a = rows_a, cols_a_rows_b
                    rows_b, cols_b = cols_a_rows_b, cols_b
                    
                    # Cria matrizes de exemplo com os mesmos tamanhos
                    A_exemplo = np.random.rand(rows_a, cols_a)
                    B_exemplo = np.random.rand(rows_b, cols_b)
                    
                    # Calcula o produto para o exemplo
                    resultado_exemplo = np.dot(A_exemplo, B_exemplo)
                    
                    salvar_relatorio_multiplicacao(
                        A=A_exemplo,
                        B=B_exemplo,
                        resultado=resultado_exemplo,
                        tempo_serial=media_serial,
                        tempo_paralelo=media_paralelo,
                        tempo_distribuido=media_distribuido if not np.isnan(media_distribuido) else None,
                        nome_arquivo=f"relatorio_caso_{i+1}.html"
                    )
                except Exception as e:
                    logger.error(f"Erro ao gerar relatório HTML: {e}", exc_info=True)
        
        # --- Análise dos Resultados ---
        logger.info("\n=== ANÁLISE DOS RESULTADOS ===")
        
        # Gera estatísticas resumidas
        if not resultados_execucao:
            logger.warning("Nenhum resultado válido foi obtido.")
            return
        
        # Cria DataFrame com os resultados
        df = pd.DataFrame(resultados_execucao)
        
        # Salva os resultados em CSV
        df.to_csv(csv_filename, index=False, float_format='%.4f')
        logger.info(f"Resultados finais salvos em '{csv_filename}'")
        
        # Exibe estatísticas
        logger.info("\nEstatísticas dos tempos de execução (segundos):")
        logger.info(f"Média Serial: {df['Serial (s)'].mean():.4f} ± {df['Serial (s)'].std():.4f}")
        logger.info(f"Média Paralelo: {df['Paralelo (s)'].mean():.4f} ± {df['Paralelo (s)'].std():.4f}")
        logger.info(f"Média Distribuído: {df['Distribuido (s)'].mean():.4f} ± {df['Distribuido (s)'].std():.4f}")
        
        logger.info("\nSpeedup médio:")
        logger.info(f"Paralelo/Serial: {df['Speedup Paralelo'].mean():.2f}x")
        logger.info(f"Distribuído/Serial: {df['Speedup Distribuido'].mean():.2f}x")
        
        # --- Geração do Gráfico ---
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), gridspec_kw={'height_ratios': [2, 1]})
            
            # Gráfico de linhas para os tempos de execução
            x = range(len(df))  # Usa índices numéricos para o eixo X
            
            # Plota os tempos de execução
            ax1.plot(x, df["Serial (s)"], marker='o', linestyle='-', label="Serial", color='#1f77b4')
            ax1.plot(x, df["Paralelo (s)"], marker='s', linestyle='--', label="Paralelo", color='#ff7f0e')
            ax1.plot(x, df["Distribuido (s)"], marker='^', linestyle='-.', label="Distribuído", color='#2ca02c')
            
            # Configurações do primeiro gráfico
            ax1.set_title('Comparação dos Tempos de Execução', fontsize=16, pad=20)
            ax1.set_ylabel('Tempo (segundos)', fontsize=12, labelpad=10)
            ax1.legend(fontsize=12, loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Adiciona rótulos de valor nos pontos
            for i, (s, p, d) in enumerate(zip(df["Serial (s)"], df["Paralelo (s)"], df["Distribuido (s)"])):
                if not np.isnan(s):
                    ax1.text(i, s + 0.1 * max(df["Serial (s)"]), f"{s:.1f}", ha='center', fontsize=8, color='#1f77b4')
                if not np.isnan(p):
                    ax1.text(i, p + 0.1 * max(df["Serial (s)"]), f"{p:.1f}", ha='center', fontsize=8, color='#ff7f0e')
                if not np.isnan(d):
                    ax1.text(i, d + 0.1 * max(df["Serial (s)"]), f"{d:.1f}", ha='center', fontsize=8, color='#2ca02c')
            
            # Configura o eixo X com os rótulos das dimensões
            ax1.set_xticks(x)
            ax1.set_xticklabels(df["Dimensoes"], rotation=45, ha='right', fontsize=8)
            
            # Gráfico de barras para o speedup
            x_speedup = np.arange(len(df))
            width = 0.35
            
            # Filtra os casos onde temos dados válidos para o speedup distribuído
            valid_dist_mask = ~np.isnan(df["Speedup Distribuido"])
            
            # Plota as barras de speedup
            rects1 = ax2.bar(x_speedup - width/2, df["Speedup Paralelo"], width, label='Paralelo/Serial', color='#ff7f0e', alpha=0.7)
            
            # Só plota as barras de speedup distribuído onde temos valores válidos
            if valid_dist_mask.any():
                rects2 = ax2.bar(x_speedup[valid_dist_mask] + width/2, 
                                df.loc[valid_dist_mask, "Speedup Distribuido"], 
                                width, 
                                label='Distribuído/Serial', 
                                color='#2ca02c', 
                                alpha=0.7)
            else:
                logger.warning("Nenhum dado de speedup distribuído válido para exibir no gráfico")
                rects2 = []
            
            # Linha de referência para speedup = 1x
            ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            
            # Configurações do segundo gráfico
            ax2.set_title('Speedup em Relação à Execução Serial', fontsize=16, pad=20)
            ax2.set_xlabel('Caso de Teste', fontsize=12, labelpad=10)
            ax2.set_ylabel('Speedup (x)', fontsize=12, labelpad=10)
            ax2.legend(fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Adiciona os valores nas barras
            def autolabel(rects, ax):
                for rect in rects:
                    try:
                        height = rect.get_height()
                        if not np.isnan(height) and height > 0:  # Só adiciona rótulo se o valor for válido e positivo
                            ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                                    f'{height:.1f}x', ha='center', va='bottom', fontsize=8, color='black')
                    except Exception as e:
                        logger.debug(f"Erro ao adicionar rótulo na barra: {e}")
            
            autolabel(rects1, ax2)
            autolabel(rects2, ax2)
            
            # Configura o eixo X com os rótulos dos casos
            ax2.set_xticks(x_speedup)
            ax2.set_xticklabels([f"Caso {i+1}" for i in x_speedup], rotation=45, ha='right', fontsize=8)
            
            # Ajusta o layout e salva a figura
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em '{plot_filename}'")
            
            # Exibe o gráfico (opcional, pode ser removido em produção)
            plt.close()
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}", exc_info=True)
        
        logger.info("\nExecução concluída com sucesso!")
        
    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")
    except Exception as e:
        logger.critical(f"Erro fatal na execução: {e}", exc_info=True)
    finally:
        # Garante que o arquivo de log seja fechado corretamente
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
    plt.show()

def formatar_matriz_para_html(matrix: np.ndarray, max_rows: int = 10, max_cols: int = 10) -> str:
    """
    Formata uma matriz para exibição em HTML, limitando o número de linhas/colunas exibidas.
    
    Args:
        matrix: Matriz a ser formatada
        max_rows: Número máximo de linhas a serem exibidas
        max_cols: Número máximo de colunas a serem exibidas
        
    Returns:
        String HTML formatada com a matriz
    """
    rows, cols = matrix.shape
    
    # Limita o número de linhas e colunas exibidas
    show_rows = min(rows, max_rows)
    show_cols = min(cols, max_cols)
    
    # Cria a tabela HTML
    html = ['<table border="1" style="border-collapse: collapse; text-align: center; margin: 10px 0;">']
    
    # Adiciona as linhas da matriz
    for i in range(show_rows):
        html.append('<tr>')
        for j in range(show_cols):
            html.append(f'<td style="padding: 5px;">{matrix[i, j]:.2f}</td>')
        
        # Adiciona reticências se a matriz for maior que o máximo exibido
        if show_cols < cols:
            html.append('<td>...</td>')
        
        html.append('</tr>')
    
    # Adiciona linha de reticências se a matriz for mais alta que o máximo exibido
    if show_rows < rows:
        html.append('<tr><td colspan="' + str(show_cols + (1 if show_cols < cols else 0)) + '" style="text-align: center;">⋮</td></tr>')
    
    html.append('</table>')
    
    # Adiciona legenda com as dimensões
    html.append('<div style="text-align: center; font-style: italic; margin-bottom: 20px;">')
    html.append(f'Matriz {rows}x{cols} (mostrando {show_rows}x{show_cols})')
    html.append('</div>')
    
    return '\n'.join(html)

def gerar_relatorio_multiplicacao(A: np.ndarray, B: np.ndarray, resultado: np.ndarray,
                                tempo_serial: float, tempo_paralelo: float, 
                                tempo_distribuido: Optional[float] = None,
                                nome_arquivo: str = "relatorio_multiplicacao.html") -> None:
    """
    Gera um relatório HTML com a multiplicação de matrizes e os tempos de execução.
    
    Args:
        A: Primeira matriz
        B: Segunda matriz
        resultado: Matriz resultado
        tempo_serial: Tempo de execução sequencial
        tempo_paralelo: Tempo de execução paralelo
        tempo_distribuido: Tempo de execução distribuído (opcional)
        
    Returns:
        String HTML com o relatório
    """
    html = ["""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Relatório de Multiplicação de Matrizes</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .matrizes { display: flex; justify-content: center; align-items: center; flex-wrap: wrap; }
            .matriz { margin: 20px; text-align: center; }
            .tempos { margin: 30px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            h1, h2 { color: #2c3e50; }
            .destaque { font-weight: bold; color: #e74c3c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Multiplicação de Matrizes</h1>
            
            <div class="matrizes">
                <div class="matriz">
                    <h3>Matriz A</h3>
    """]
    
    # Adiciona a matriz A
    html.append(formatar_matriz_para_html(A))
    
    # Adiciona o sinal de multiplicação
    html.append("""
                </div>
                <div style="font-size: 24px; margin: 0 20px;">×</div>
                <div class="matriz">
                    <h3>Matriz B</h3>
    """)
    
    # Adiciona a matriz B
    html.append(formatar_matriz_para_html(B))
    
    # Adiciona o sinal de igual
    html.append("""
                </div>
                <div style="font-size: 24px; margin: 0 20px;">=</div>
                <div class="matriz">
                    <h3>Resultado</h3>
    """)
    
    # Adiciona a matriz resultado
    html.append(formatar_matriz_para_html(resultado))
    
    # Adiciona os tempos de execução
    html.append("""
                </div>
            </div>
            
            <div class="tempos">
                <h2>Tempos de Execução</h2>
                <ul>
    """)
    
    html.append(f'<li><span class="destaque">Serial:</span> {tempo_serial:.6f} segundos</li>')
    html.append(f'<li><span class="destaque">Paralelo:</span> {tempo_paralelo:.6f} segundos (Speedup: {tempo_serial/tempo_paralelo:.2f}x)</li>')
    
    if tempo_distribuido is not None:
        html.append(f'<li><span class="destaque">Distribuído:</span> {tempo_distribuido:.6f} segundos (Speedup: {tempo_serial/tempo_distribuido if tempo_distribuido > 0 else 0:.2f}x)</li>')
    
    html.append("""
                </ul>
            </div>
        </div>
    </body>
    </html>
    """)
    
    return '\n'.join(html)

def salvar_relatorio_multiplicacao(A: np.ndarray, B: np.ndarray, resultado: np.ndarray,
                                tempo_serial: float, tempo_paralelo: float, 
                                tempo_distribuido: Optional[float] = None,
                                nome_arquivo: str = "relatorio_multiplicacao.html") -> Path:
    """
    Gera e salva um relatório HTML com a multiplicação de matrizes.
    
    Args:
        A: Primeira matriz
        B: Segunda matriz
        resultado: Matriz resultado
        tempo_serial: Tempo de execução sequencial
        tempo_paralelo: Tempo de execução paralelo
        tempo_distribuido: Tempo de execução distribuído (opcional)
        nome_arquivo: Nome do arquivo de saída (será salvo em html_reports/)
        
    Returns:
        Path: Caminho completo para o arquivo salvo
    """
    # Gera o conteúdo HTML
    conteudo = gerar_relatorio_multiplicacao(A, B, resultado, tempo_serial, tempo_paralelo, tempo_distribuido)
    
    # Define o caminho completo do arquivo
    filepath = HTML_REPORTS_DIR / nome_arquivo
    
    # Garante que o diretório existe
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Salva em arquivo
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(conteudo)
    
    logger.info(f"Relatório salvo em: {filepath}")
    return filepath

if __name__ == "__main__":
    try:
        start_time = time.perf_counter()
        main()
    except Exception as e:
        logger.critical(f"Erro não tratado: {e}", exc_info=True)
    finally:
        end_time = time.perf_counter()
        logger.info(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")
        
        # Encerra corretamente os recursos de logging
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
