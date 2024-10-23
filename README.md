# fringe_projection
Algoritmo para projeção de franjas

**status:** concluído

### Pré-requisitos
Necessário ter um editor para trabalhar com o código em python como o [PyCharm](https://www.jetbrains.com/pycharm/)

### Instruções de instalação
- Clonar este repositório no PyCharm

### Execução
Após a instalação, você obterá o ambiente de desenvolvimento. O código principal responsável pela saída do algoritmo é o arquivo 'main'. 
Os outros arquivos encontrados na pasta 'include' correspondem às bibliotecas montadas para o funcionamento da lógica de todo o código executado no 'main'. 
Dessa maneira, todos os parâmetros de entrada são passados neste mesmo arquivo.

**Inicializando os parametros na main**

O primeiro passo é inicializar as câmeras. Nesse passo, é preciso informar o número de série de ambas as câmeras. 
Para isso, na linha 23 do código, é chamada a biblioteca 'StereoCameraController', onde se deve configurar o número de série das câmeras, garantindo que o número de série informado corresponda, respectivamente, à câmera da esquerda e à da direita.

```python
    stereo_ctrl = StereoCameraController(left_serial=16378750, right_serial=16378734)
```
Finalizada essa etapa, o próximo parâmetro do código é o arquivo YAML que corresponde ao resultado das calibrações e pode ser alterado na linha 99 do código, modificando apenas o caminho do arquivo.
````python
    # read the yaml_file
    yaml_file = '/home/daniel/PycharmProjects/fringe_projection/params/20241018_bouget.yaml'
````

O último parâmetro que pode ser alterado são os limites dos pontos x, y e z, que afetarão a quantidade de pontos, o tamanho e a região da imagem onde os pontos serão plotados. Você pode verificar isso na linha 109 do código.
````python
    # Inverse Triangulation for Fringe projection
    zscan = InverseTriangulation(yaml_file)
    zscan.points3d(x_lim=(-250, 500), y_lim=(-100, 400), z_lim=(-200, 200), xy_step=7, z_step=0.1, visualize=False)
````

**Rodando a main**

Com os passos anteriores verificados, você conseguirá rodar o código e, como resultado, deverá obter a nuvem de pontos do objeto projetado em um arquivo .csv, além do gráfico da nuvem de pontos

### Sobre a implementação do código
Para a criação do algoritmo de projeção de franjas, foram desenvolvidas algumas bibliotecas, nas quais cada uma é responsável por implementar funcionalidades no código principal. 
É necessária a compreensão de cada uma delas.

**FringePattern:**
Biblioteca responsável por criar e obter as imagens de franjas senoidais.

**GrayCode:**
Biblioteca responsável por criar e obter as imagens de código Gray. Outra funcionalidade desse código é converter uma lista de inteiros em suas representações binárias no código Gray e também ordenar os valores do código Gray de acordo com sua posição na imagem.

**stereo_fringe_process:**
Biblioteca responsável por todo o processamento das imagens capturadas, realizando os devidos cálculos para a obtenção do mapa de fase absoluto.

**InverseTriangulation:**
Biblioteca responsável por criar a nuvem de pontos. Todas as funções nesse módulo foram baseadas no método de triangulação inversa.