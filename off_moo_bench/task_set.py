SyntheticFunction = [
    "ZDT1-Exact-v0",
    "ZDT2-Exact-v0",
    "ZDT3-Exact-v0",
    "ZDT4-Exact-v0",
    "ZDT6-Exact-v0",
    "OmniTest-Exact-v0",
    "VLMOP1-Exact-v0",
    "VLMOP2-Exact-v0",
    "VLMOP3-Exact-v0",
    "DTLZ1-Exact-v0",
    "DTLZ2-Exact-v0",
    "DTLZ3-Exact-v0",
    "DTLZ4-Exact-v0",
    "DTLZ5-Exact-v0",
    "DTLZ6-Exact-v0",
    "DTLZ7-Exact-v0",
]

MONASSequence = [
    "C10MOP1-Exact-v0",
    "C10MOP2-Exact-v0",
    "C10MOP3-Exact-v0",
    "C10MOP4-Exact-v0",
    "C10MOP5-Exact-v0",
    "C10MOP6-Exact-v0",
    "C10MOP7-Exact-v0",
    "C10MOP8-Exact-v0",
    "C10MOP9-Exact-v0",
    "IN1KMOP1-Exact-v0",
    "IN1KMOP2-Exact-v0",
    "IN1KMOP3-Exact-v0",
    "IN1KMOP4-Exact-v0",
    "IN1KMOP5-Exact-v0",
    "IN1KMOP6-Exact-v0",
    "IN1KMOP7-Exact-v0",
    "IN1KMOP8-Exact-v0",
    "IN1KMOP9-Exact-v0",
]

MONASLogits = [
    "NASBench201Test-Exact-v0",
]

MONAS = MONASSequence + MONASLogits

MOCOPermutation = [
    "BiTSP20-Exact-v0",
    "BiTSP50-Exact-v0",
    "BiTSP100-Exact-v0",
    "BiTSP500-Exact-v0",
    "TriTSP20-Exact-v0",
    "TriTSP50-Exact-v0",
    "TriTSP100-Exact-v0",
    "BiCVRP20-Exact-v0",
    "BiCVRP50-Exact-v0",
    "BiCVRP100-Exact-v0",
    "BiKP50-Exact-v0",
    "BiKP100-Exact-v0",
    "BiKP200-Exact-v0",
]

MOCOContinuous = [
    "Portfolio-Exact-v0"
]

MOCO = MOCOPermutation + MOCOContinuous

MORL = [
    "MOSwimmerV2-Exact-v0", 
    "MOHopperV2-Exact-v0",
]

ScientificDesignContinuous = [
    "Molecule-Exact-v0",
]

ScientificDesignSequence = [
    "Regex-Exact-v0",
    "ZINC-Exact-v0",
    "RFP-Exact-v0",
]

ScientificDesign = ScientificDesignContinuous + ScientificDesignSequence

RESuite = [
    "RE21-Exact-v0",
    "RE22-Exact-v0",
    "RE23-Exact-v0",
    "RE24-Exact-v0",
    "RE25-Exact-v0",
    "RE31-Exact-v0",
    "RE32-Exact-v0",
    "RE33-Exact-v0",
    "RE34-Exact-v0",
    "RE35-Exact-v0",
    "RE36-Exact-v0",
    "RE37-Exact-v0",
    "RE41-Exact-v0",
    "RE42-Exact-v0",
    "RE61-Exact-v0",
]

ALLTASKS = SyntheticFunction + MONAS + MOCO + MORL + ScientificDesign + RESuite