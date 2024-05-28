import random

# Define el nombre del archivo de entrada y salida
input_filename = '/home/xnmaster/XNAPproject-grup-04/data/spa.txt'
output_filename = '/home/xnmaster/XNAPproject-grup-04/data/spa_sample.txt'

# Número de líneas a seleccionar
num_lines_to_select = 20000

# Lee todas las líneas del archivo
with open(input_filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Selecciona 20,000 líneas aleatorias
selected_lines = random.sample(lines, num_lines_to_select)

# Escribe las líneas seleccionadas en un nuevo archivo
with open(output_filename, 'w', encoding='utf-8') as file:
    file.writelines(selected_lines)

print(f'{num_lines_to_select} líneas aleatorias han sido guardadas en {output_filename}')
