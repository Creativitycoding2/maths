from django.shortcuts import render
from .tes4 import parse_equation, balance_equation
from sympy import symbols
# Create your views here.


def assemble_equation(reactants, products, solution):
    try:
        # Get the number of coefficients
        num_coefficients = len(solution)

        # If the last coefficient is not defined, add it to the solution dictionary
        solution[symbols(f'x{num_coefficients}')] = symbols(f'x{num_coefficients}')
        print(num_coefficients)
        print(solution)
        # Initialize empty strings for reactants and products sides
        reactants_side = ""
        products_side = ""

        # Assemble reactants side of the equation
        for i, compound in enumerate(reactants):
            reactant_str = "".join(
                [f"{element}{count}" if count != 1 else f"{element}" for element, count in compound.items()])
            reactants_side += f"{solution[symbols(f'x{i}')]}{reactant_str} + "

            # Assemble products side of the equation
        for i, compound in enumerate(products):
            product_str = "".join(
                [f"{element}{count}" if count != 1 else f"{element}" for element, count in compound.items()])
            terms = f"{solution[symbols(f'x{i + len(reactants)}')]}{product_str}"
            if terms:
                products_side += terms + " + "

            # Construct the chemical equation
        a = symbols(f'x{num_coefficients}')
        equation = f"{reactants_side[:-3]} = {products_side[:-3]}".replace(f'{a}', '1')
        return equation

    except KeyError:
        equation = "couldn't balance equation"
        return equation


def num1(request):
    return render(request, 'raw.html')


def num2(request):
    equation = request.GET.get('hero-field', 'default')
    compounds = parse_equation(equation)
    reactants = compounds['reactants']
    products = compounds['products']
    solution = balance_equation(reactants, products)
    assemble = assemble_equation(reactants, products, solution[0])
    modify = solution[1].replace(")", "\n").replace("Eq(", "--> ").replace(",", " =")
    params = {"string": assemble, "equation": modify}

    return render(request, 'num.html', params)
