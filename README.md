# Multiplicação de Matrizes Distribuída

Este projeto implementa um sistema de multiplicação de matrizes que pode ser executado de três maneiras diferentes: serial, paralela e distribuída. O objetivo é comparar o desempenho entre essas abordagens.

## 🚀 Funcionalidades

- Multiplicação de matrizes em modo serial (um único processo)
- Multiplicação paralela usando multiprocessamento
- Multiplicação distribuída entre múltiplos servidores
- Geração de relatórios em HTML com os resultados
- Gráficos comparativos de desempenho

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

Em terminais separados, execute:

```bash
# Servidor 1
python server.py --port 12345

# Servidor 2 (opcional, para modo distribuído)
python server.py --port 12346
```

### 2. Executando o Cliente

```bash
python client.py
```

### 3. Opções de Execução

O cliente suporta os seguintes argumentos:

- `--servers`: Lista de servidores no formato "host:porta" (padrão: "localhost:12345,localhost:12346")
- `--test-cases`: Número de casos de teste (padrão: 3)
- `--runs`: Número de execuções por caso de teste (padrão: 2)

Exemplo:
```bash
python client.py --servers "localhost:12345,localhost:12346" --test-cases 5 --runs 3
```

## 📊 Saída

O programa gera os seguintes arquivos:

- `resultados_comparativos.csv`: Dados brutos dos tempos de execução
- `comparativo_execucao.png`: Gráfico comparativo de desempenho
- `html_reports/`: Pasta contendo relatórios detalhados em HTML
- `logs/`: Arquivos de log com informações detalhadas da execução

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

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## ✉️ Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter) - seu.email@exemplo.com

Link do Projeto: [https://github.com/Ravi-F/Av3Paralela](https://github.com/Ravi-F/Av3Paralela)

---

<div align="center">
    Feito com ❤️ por Você!
</div>
