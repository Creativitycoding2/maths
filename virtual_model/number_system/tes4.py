import re

from sympy import Eq, solve, symbols


def parse_equation(equation_):
    equation_sides = equation_.split('=')
    reactants_ = equation_sides[0].strip()
    products_ = equation_sides[1].strip()
    subset = {'reactants': [], 'products': []}

    for side in [reactants_, products_]:
        compounds_ = side.strip().split('+')
        for compound in compounds_:
            elements_ = {}
            parts = re.findall(r'([A-Z][a-z]*)(\d*)', compound)
            for element, count in parts:
                count = int(count) if count else 1
                elements_[element] = elements_.get(element, 0) + count
            if side == reactants_:
                subset['reactants'].append(elements_)
            else:
                subset['products'].append(elements_)

    return subset


def balance_equation(reactants__, products__):
    # Collect all unique elements
    print(reactants__, products__)
    all_elements = set().union(*(reactants__ + products__))
    print(all_elements)
    # Setup symbols for coefficients
    coefficients = symbols(' '.join(['x{}'.format(i) for i in range(len(reactants__) + len(products__))]))

    # Setup equations for each element
    equations = []
    empty_str = ""
    for element in all_elements:
        reactant_sum = sum([compound.get(element, 0) * coefficients[i] for i, compound in enumerate(reactants__)])
        product_sum = sum(
            [compound.get(element, 0) * coefficients[i + len(reactants__)] for i, compound in enumerate(products__)])
        eq = Eq(reactant_sum, product_sum)
        empty_str += str(eq)
        equations.append(eq)

    # Solve equations
    solution_ = [solve(equations, coefficients), empty_str]
    print("Solutions:", solution_)

    return solution_


if __name__ == '__main__':
    cont = True
    while cont:
        try:
            equation = input("Enter the chemical equation: ")
            compounds = parse_equation(equation)
            reactants = compounds['reactants']
            products = compounds['products']
            solution = balance_equation(reactants, products)
            print("Balanced coefficients:", solution)
        except Exception as e:
            print(e)
        cont = True if "yes" in input("Do you want to continue? (yes/no): ").lower() else False
