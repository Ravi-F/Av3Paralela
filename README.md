# Multiplicação de Matrizes Distribuída

Este projeto implementa um sistema de multiplicação de matrizes que pode ser executado de três maneiras diferentes: serial, paralela e distribuída. O objetivo principal é demonstrar como a distribuição de tarefas entre múltiplos servidores pode melhorar o desempenho de operações matriciais intensivas.

## 🔄 Fluxo de Execução

### 1. Geração das Matrizes
- O cliente gera duas matrizes A e B de tamanhos configuráveis
- A matriz A é dividida em submatrizes para processamento distribuído

### 2. Processamento Distribuído
- Cada servidor recebe uma parte da matriz A e a matriz B completa
- Os cálculos são realizados em paralelo nos servidores
- Os resultados parciais são retornados ao cliente

### 3. Consolidação dos Resultados
- O cliente recebe e combina os resultados parciais
- A matriz resultante C é formada pela concatenação das partes processadas
- Relatórios e métricas de desempenho são gerados

## 🚀 Funcionalidades

### Modos de Operação
- **Serial**: Processamento sequencial em um único núcleo
- **Paralelo**: Multiprocessamento local utilizando todos os núcleos disponíveis
- **Distribuído**: Cálculos distribuídos entre múltiplos servidores via sockets

### Características
- Divisão automática da carga de trabalho
- Comunicação assíncrona entre cliente e servidores
- Tolerância a falhas com sistema de retentativas
- Geração de relatórios detalhados em HTML
- Análise comparativa de desempenho entre os modos de execução

## 📋 Pré-requisitos

- Python 3.7 ou superior
- Bibliotecas Python:
  - numpy
  - matplotlib
  - pandas
  - colorama

## 🛠️ Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/Ravi-F/Av3Paralela.git
   cd Av3Paralela
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Como Executar

### 1. Iniciando os Servidores

Para o modo distribuído, é necessário iniciar pelo menos um servidor. Para melhor desempenho, recomenda-se pelo menos dois servidores:

```bash
# Terminal 1 - Primeiro servidor
python server.py --port 12345

# Terminal 2 - Segundo servidor (opcional)
python server.py --port 12346
```

Cada servidor pode processar partes independentes da matriz A em paralelo, acelerando significativamente o processamento para matrizes grandes.

### 2. Executando o Cliente

```bash
python client.py
```

### 3. Opções de Execução

O cliente oferece várias opções para personalizar a execução:

- `--servers`: Especifica os servidores no formato "host:porta" (padrão: "localhost:12345,localhost:12346")
- `--test-cases`: Número de casos de teste com diferentes tamanhos de matriz (padrão: 3)
- `--runs`: Número de execuções por caso de teste para cálculo de médias (padrão: 2)
- `--min-size`: Tamanho mínimo das matrizes (padrão: 100)
- `--max-size`: Tamanho máximo das matrizes (padrão: 1000)

Exemplo de execução com parâmetros personalizados:
```bash
python client.py --test-cases 5 --runs 3 --min-size 50 --max-size 500
```

Exemplo:
```bash
python client.py --servers "localhost:12345,localhost:12346" --test-cases 5 --runs 3
```

## 📊 Saída e Análise

### Arquivos Gerados
- `resultados_comparativos.csv`: Dados brutos de tempos de execução e speedup
- `comparativo_execucao.png`: Gráfico comparativo entre os modos de execução
- `html_reports/`: Relatórios HTML detalhados para cada caso de teste
  - Visualização das matrizes de entrada e saída
  - Métricas de desempenho detalhadas
  - Comparação entre os modos de execução
- `logs/`: Registros detalhados para diagnóstico

### Análise de Desempenho
O sistema calcula automaticamente:
- Tempo total de execução para cada modo
- Speedup em relação à execução serial
- Eficiência da paralelização
- Uso de recursos

## 🏗️ Estrutura do Projeto

```
Av3Paralela/
├── client.py            # Cliente principal
├── server.py            # Servidor de processamento
├── requirements.txt     # Dependências do projeto
├── .gitignore          # Arquivos ignorados pelo Git
├── logs/               # Arquivos de log
├── html_reports/       # Relatórios em HTML
├── resultados_comparativos.csv  # Dados de execução
└── comparativo_execucao.png    # Gráfico de desempenho
```

## 🤝 Como Contribuir

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das suas alterações (`git commit -m 'Add some AmazingFeature'`)
4. Faça push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

