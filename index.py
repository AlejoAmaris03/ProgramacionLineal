import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pulp as p
from flask import Flask, render_template, request, send_file, url_for
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    # Obtener los valores de x
    x1_coef = float(request.form['x1'])
    x2_coef = float(request.form['x2'])
    
    # Obtener la función objetivo
    model = p.LpProblem("PL_Max",p.LpMaximize)
    
    #Crear variables
    x1 = p.LpVariable("x1", lowBound=0, cat=p.LpInteger)
    x2_1 = p.LpVariable("x2", lowBound=0, cat=p.LpInteger)
    
    # Obtener el número de restricciones
    num_constraints = int(request.form['num_constraints'])
    constraints = []
    
    for i in range(num_constraints):
        a1 = float(request.form[f'constraint_{i}_a1'])
        a2 = float(request.form[f'constraint_{i}_a2'])
        sign = str(request.form[f'restrictionSign_{i}'])
        b = float(request.form[f'constraint_{i}_b'])

        constraints.append((a1, a2, sign, b))
      
    # Añadir la función objetivo
    model += x1_coef*x1 + x2_coef*x2_1, "Función Objetivo"
    
    # Añadir las restricciones
    for (a1, a2, sign, b) in constraints:
        if sign == "<=":
            model += a1*x1 + a2*x2_1 <= b
        else:
            model += a1*x1 + a2*x2_1 >= b
    
    # Resolver el problema de optimización
    model.solve()

    # Definir el rango de valores de x para graficar
    x_range_limit = int(request.form['x_range_limit'])
    x_range_points = int(request.form["x_range_points"])
    x = np.linspace(0, x_range_limit, x_range_points)

    # Graficar las restricciones
    plt.plot(0,0,label="Z = "f"{x1_coef}""x1 + "f"{x2_coef}""x2")
    
    for (a1, a2, sign, b) in constraints:
        if sign == "<=":
            if(a2 == 0):
                plt.axvline(b/a1, color='violet', linestyle='-', label=f'{a1}x1 + {a2}x2 <= {b}')
                plt.fill_between(x, 0, x2, where=((x2 >= 0) & (x >= 0)), alpha=0.1) 
            else:   
                x2 = (b - (a1 * x)) / a2 
                plt.plot(x, x2, label=f'{a1}x1 + {a2}x2 <= {b}')       
                plt.fill_between(x, 0, x2, where=((x2 >= 0) & (x >= 0)), alpha=0.1)    
        elif sign == ">=":            
            if(a2 == 0):
                plt.axvline(b/a1, color='violet', linestyle='-', label=f'{a1}x1 + {a2}x2 >= {b}')
                plt.fill_between(x, 0, x2, where=((x2 >= b) & (x >= b)), alpha=0.1)      
            else:    
                x2 = (b - (a1 * x)) / a2
                plt.plot(x, x2, label=f'{a1}x1 + {a2}x2 >= {b}') 
                plt.fill_between(x, 0, x2, where=((x2 >= 0) & (x >= 0)), alpha=0.1)    
        
    # Graficar la función objetivo
    plt.plot(x1.varValue, x2_1.varValue, 'ro', markersize=10)
    plt.title("Método Gráfico de Programación Lineal\n x1 = "f"{x1.varValue}""; x2 = "f"{x2_1.varValue}""; Z = "f"{p.value(model.objective)}""")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axhline(0, color='green', linewidth=1)
    plt.axvline(0, color='red', linewidth=1)
    plt.grid(color="black", linestyle="--", linewidth=0.5)
    plt.legend()
    
    # Guardar el gráfico en un archivo temporal
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)