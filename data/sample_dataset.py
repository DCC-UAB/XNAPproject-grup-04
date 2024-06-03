import random


"""
# Proporciona la ruta al archivo de entrada
input_filename = 'data/spa.txt'
output_filename = 'data/spa_sample_frases_largas.txt'

# Número de líneas a seleccionar
num_lines_to_select = 70000

# Función para contar palabras en una línea
def count_words(line):
    return len(line.split())

# Lee todas las líneas del archivo
with open(input_filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Filtra las líneas que tienen menos de 8 palabras
filtered_lines = [line for line in lines if count_words(line) >=11]

# Verifica que hay suficientes líneas para seleccionar
if len(filtered_lines) < num_lines_to_select:
    raise ValueError("No hay suficientes líneas con menos de 6 palabras.")

# Selecciona 20,000 líneas aleatorias de las líneas filtradas
selected_lines = random.sample(filtered_lines, num_lines_to_select)

# Escribe las líneas seleccionadas en un nuevo archivo
with open(output_filename, 'w', encoding='utf-8') as file:
    file.writelines(selected_lines)

print(f'{num_lines_to_select} líneas con menos de 6 palabras han sido guardadas en {output_filename}')
"""




# Define la ruta del archivo de entrada y de salida
input_filename = 'data/cat.txt'
output_filename = 'data/cat_sample.txt'

# Lee las líneas del archivo de entrada
with open(input_filename, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Desordena las líneas
random.shuffle(lines)

# Escribe las líneas desordenadas en el archivo de salida
with open(output_filename, "w", encoding="utf-8") as file:
    file.writelines(lines)

print("El archivo ha sido desordenado exitosamente.")