# ae-plcg
This project is a Python3 implementation of Probabilistic Left Corner Grammars (PLCGs; Manning and Carpenter, 1997) including Arc-Eager PLCGs (AE-PLCG) and simple evaluation tools as detailed in (Barnett, 2021).

## Abstract
Psycholinguistic research has posited arc-eager left corner parsing as a psychologically viable candidate for the human parsing mechanism (Resnik, 1992). Using probabilistic left-corner grammars (PLCGs), as introduced by Manning and Carpenter (1997), as a testbed, this thesis examines the probabilistic mechanisms involved in arc-eager tree construction. By moving attachment decisions earlier in the decision tree, arc-eager PLCGs gain probabilistic advantage over their arc-standard counterparts due to recursive left-corner embeddings, tree productions of the form `A -> Aγ`, which are abundamt in datasets like the Penn Treebank (PTB), and many would argue have psychological reality. This advantage is fully independent of the well-documented stack depth limit advantage in arc-eager construction of right-branching structures.  

## Requirements
This project imports the following external modules, both of which can be attained using standard methods including `pip`: 
- [NLTK](https://www.nltk.org/install.html)
- [NumPy](https://numpy.org/install/)

This project also imports the following packages, which are generally pre-installed with Python:
- [collections](https://docs.python.org/3/library/collections.html)
- [re](https://docs.python.org/3/library/re.html)
- [json](https://docs.python.org/3/library/json.html)
- [time](https://docs.python.org/3/library/time.html)

## Model Training Data Variants
- PLCG-F  - full PTB
- PLCG-D  - de-lexicalized PTB
- PLCG-UC - unary collapsed PTB
- PLCG-BR - right-binarized PTB
- PLCG-BL - left-binarized PTB
- PLCG-RX - recursive left-corner removed PTB


## References
Barnett (2021) - Arc-Eager Construction Provides Learning Advantage Beyond Stack Management

Manning and Carpenter (1997) - [Probabilistic Parsing Using Left Corner Language Models](https://arxiv.org/pdf/cmp-lg/9711003.pdf)

Resnik (1992) - [Left-Corner Parsing and Psychological Plausability](https://www.aclweb.org/anthology/C92-1032.pdf)
