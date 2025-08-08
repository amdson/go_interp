from collections import defaultdict
aas = "ACDEFGHIKLMNPQRSTVWY"

aromatic_aas = "YWF"
aromatic_scale = defaultdict(float, {rr: int(rr in aromatic_aas) for rr in aas})

positive_pKs = {"K": 10.0, "R": 12.0, "H": 5.98}
negative_pKs = {"D": 4.05, "E": 4.45, "C": 9.0, "Y": 10.0}
# pH = 6.8
# pKs_scale = {aa: (1.0 / (10 ** (pH - pK) + 1.0)) for aa, pK in positive_pKs.items()}
# pKs_scale.update({aa: (-1.0 / (10 ** (pH - pK) + 1.0)) for aa, pK in negative_pKs.items()})
pKs_scale = {aa: pK for aa, pK in positive_pKs.items()}
pKs_scale.update({aa: -pK for aa, pK in negative_pKs.items()})
pKs_scale = {aa: (pKs_scale[aa] if aa in pKs_scale else 0) for aa in aas}
pKs_scale = defaultdict(float, pKs_scale)

gravy_scale = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
                    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
                    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
                    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
gravy_scale = defaultdict(lambda: sum(gravy_scale.values())/len(gravy_scale), 
                                   gravy_scale)

protein_weight_scale = {
    "A": 89.0932,
    "C": 121.1582,
    "D": 133.1027,
    "E": 147.1293,
    "F": 165.1891,
    "G": 75.0666,
    "H": 155.1546,
    "I": 131.1729,
    "K": 146.1876,
    "L": 131.1729,
    "M": 149.2113,
    "N": 132.1179,
    "O": 255.3134,
    "P": 115.1305,
    "Q": 146.1445,
    "R": 174.201,
    "S": 105.0926,
    "T": 119.1192,
    "U": 168.0532,
    "V": 117.1463,
    "W": 204.2252,
    "Y": 181.1885,
}
protein_weight_scale = defaultdict(lambda: sum(protein_weight_scale.values())/len(protein_weight_scale), 
                                   protein_weight_scale)
