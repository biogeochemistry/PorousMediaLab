
#! the example is taken from stackoverflow:
#! https://stackoverflow.com/questions/45220032/how-to-balance-a-chemical-equation-in-python-2-7-using-matrices

# Find minimum integer coefficients for a chemical reaction like
#   A * NaOH + B * H2SO4 -> C * Na2SO4 + D * H20
import sympy
import re

# match a single element and optional count, like Na2
ELEMENT_CLAUSE = re.compile("([A-Z][a-z]?)([0-9]*)")

def parse_compound(compound):
    """
    Given a chemical compound like Na2SO4,
    return a dict of element counts like {"Na":2, "S":1, "O":4}
    """
    assert "(" not in compound, "This parser doesn't grok subclauses"
    return {el: (int(num) if num else 1) for el, num in ELEMENT_CLAUSE.findall(compound)}

# def main():
#     print("\nPlease enter left-hand list of compounds, separated by spaces:")
#     lhs_strings = input().split()
#     lhs_compounds = [parse_compound(compound) for compound in lhs_strings]

#     print("\nPlease enter right-hand list of compounds, separated by spaces:")
#     rhs_strings = input().split()
#     rhs_compounds = [parse_compound(compound) for compound in rhs_strings]

#     # Get canonical list of elements
#     els = sorted(set().union(*lhs_compounds, *rhs_compounds))
#     els_index = dict(zip(els, range(len(els))))

#     # Build matrix to solve
#     w = len(lhs_compounds) + len(rhs_compounds)
#     h = len(els)
#     A = [[0] * w for _ in range(h)]
#     # load with element coefficients
#     for col, compound in enumerate(lhs_compounds):
#         for el, num in compound.items():
#             row = els_index[el]


from enum import Enum
from re import findall


class Element(Enum):
    H = ("Hydrogen", 1)
    He = ("Helium", 2)
    Li = ("Lithium", 3)
    Be = ("Beryllium", 4)
    B = ("Boron", 5)
    C = ("Carbon", 6)
    N = ("Nitrogen", 7)
    O = ("Oxygen", 8)
    F = ("Fluorine", 9)
    Ne = ("Neon", 10)
    Na = ("Sodium", 11)
    Mg = ("Magnesium", 12)
    Al = ("Aluminium", 13)
    Si = ("Silicon", 14)
    P = ("Phosphorus", 15)
    S = ("Sulfur", 16)
    Cl = ("Chlorine", 17)
    Ar = ("Argon", 18)
    K = ("Potassium", 19)
    Ca = ("Calcium", 20)
    Sc = ("Scandium", 21)
    Ti = ("Titanium", 22)
    V = ("Vanadium", 23)
    Cr = ("Chromium", 24)
    Mn = ("Manganese", 25)
    Fe = ("Iron", 26)
    Co = ("Cobalt", 27)
    Ni = ("Nickel", 28)
    Cu = ("Copper", 29)
    Zn = ("Zinc", 30)
    Ga = ("Gallium", 31)
    Ge = ("Germanium", 32)
    As = ("Arsenic", 33)
    Se = ("Selenium", 34)
    Br = ("Bromine", 35)
    Kr = ("Krypton", 36)
    Rb = ("Rubidium", 37)
    Sr = ("Strontium", 38)
    Y = ("Yttrium", 39)
    Zr = ("Zirconium", 40)
    Nb = ("Niobium", 41)
    Mo = ("Molybdenum", 42)
    Tc = ("Technetium", 43)
    Ru = ("Ruthenium", 44)
    Rh = ("Rhodium", 45)
    Pd = ("Palladium", 46)
    Ag = ("Silver", 47)
    Cd = ("Cadmium", 48)
    In = ("Indium", 49)
    Sn = ("Tin", 50)
    Sb = ("Antimony", 51)
    Te = ("Tellurium", 52)
    I = ("Iodine", 53)
    Xe = ("Xenon", 54)

    def get_name(self):
        return self.value[0]

    def get_atomic_num(self):
        return self.value[1]

    def get_symbol(self):
        return self.name

    def __repr__(self):
        return self.get_symbol()

#! code taken from
#! https://github.com/jakegoodman01/Chemical-Equation-Balancer/blob/master/equation_balancer.py

class Chemical:
    def __init__(self, formula):
        self._elements = findall("[A-Z][^A-Z]*", formula)
        self._element_freq = {}
        self.formula = formula

        # Making frequency dictionary for elements
        atoms_in_parens = []
        poly_ion_freq = None
        atom_is_in_parens = False
        for i in range(len(self._elements)):
            element = self._elements[i]
            # If the current element has a left paren, I must keep track of the coming elements
            if "(" in element:
                atom_is_in_parens = True
                element = self._elements[i][:-1]
                self._elements[i] = element
            # If the current element has a right paren, the polyatomic ion contains all previous
            # elements up until the last left paren
            elif ")" in element:
                poly_ion_freq = int(element[element.index(")") + 2:])
                element = self._elements[i][:element.index(")")]
                self._elements[i] = element

            # If the current element has a caret symbol, I must note its frequency
            if "^" in element:
                freq = int(element[element.index("^") + 1:])
                element = element[:element.index("^")]
            else:
                freq = 1

            # If the current element is within parens, I must keep track of it
            if atom_is_in_parens and "(" not in element:
                atoms_in_parens.append(element)

            if element in self._element_freq:
                self._element_freq[element] += freq
            else:
                self._element_freq[element] = freq

            # If I have finished noting the polyatomic ion, I multiply each atom
            # in that ion according to its frequency
            if poly_ion_freq is not None:
                for e in atoms_in_parens[1:]:
                    self._element_freq[e] *= poly_ion_freq

        # Re-writing self.elements with only the element symbol
        self._elements = []
        for item in self._element_freq.items():
            for i in range(item[1]):
                self._elements.append(item[0])

        # Making self.elements_obj to store actual Element objects instead of strings
        self.elements_obj = []
        for element_name in self._elements:
            if hasattr(Element, element_name):
                self.element = getattr(Element, element_name)
            else:
                raise RuntimeError(f'Unknown element name {element_name}')
            self.elements_obj.append(self.element)

    def __repr__(self):
        return self.formula

    def get_elements(self):
        return self.elements_obj


class Reactants:
    def __init__(self, *chemicals):
        self._chemicals = chemicals
        self._elements, self._element_freq = Reactants.map_chemicals(chemicals)

    @classmethod
    def map_chemicals(cls, chemicals):
        """
        :param chemicals: A tuple of n chemicals
        :return: array of all of the elements, dictionary of every element to its frequency
        """
        elements = []
        for chemical in chemicals:
            for element in chemical.get_elements():
                elements.append(element)

        element_freq = {}
        for element in elements:
            if element in element_freq:
                element_freq[element] += 1
            else:
                element_freq[element] = 1

        return elements, element_freq

    def __repr__(self):
        """
        Repr method for Reactants
        :return: The elements written as a sum
        """
        output = str(self._chemicals[0])
        for chemical in self._chemicals[1:]:
            output += f" + {chemical}"
        return output

    def get_elements(self):
        return self._elements

    def get_frequency(self):
        return self._element_freq

    def get_chemicals(self):
        return self._chemicals


def balance_equation(r1, r2):
    # If the reactants contain different elements, a ValueError is raised
    if set(r1.get_elements()) != set(r2.get_elements()):
        raise ValueError(f"[{r1}] will never become [{r2}]")

    for element in r1.get_frequency():
        # If the frequency of element is not the same
        if r1.get_frequency()[element] != r2.get_frequency()[element]:
            print(f"{element.get_name()}s are not the same!")
        else:
            print(f"{element.get_name()}s are the same!")


c1 = Chemical("BeSO^4")
c2 = Chemical("CaO")
reactant1 = Reactants(c1, c2)

c3 = Chemical("Ca(SO^4)^3")
c4 = Chemical("H(BeO)^2")
reactant2 = Reactants(c3, c4)

#be = balance_equation(reactant1, reactant2)
print(c4._element_freq)