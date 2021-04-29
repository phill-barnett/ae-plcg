from nltk.tree import Tree
from nltk.treetransforms import *
from collections import defaultdict
import numpy as np
import re
import json
import time

### Dendrology ###

def crack(tree): # (nltk.Tree) -> nltk.Tree -- this can be accomplished with the remove_empty_top_bracketing flag in Tree.fromstring(), but crack allows us this ability post-nltk.Tree-ification
    if tree.label() == '':
        return tree[0]
    else:
        return tree

def read_node(n): # removes indices from nodes and ensures uppercase
    n = re.sub(r"([^\d\s])((-|=)\d+)?", r"\1", n)
    return n.upper()

def read_leaf(l): # removes indices from leaves and forces lowercase
    l = re.sub(r"([^\d\s])((-|=)\d+)?", r"\1", l)
    #l = 'w'
    return l.lower()

def read_leaf_lexless(l): # replaces all leaf symbols with an 'x' to remove lexical information from the tree - now we're just comparing structure
    l = 'w'
    return l

def read_tree(t): # so i can reuse the same fromstring settings without retyping each flag
        t = Tree.fromstring(
            t, 
            read_node=(lambda n: read_node(n)),  
            read_leaf=(lambda l: read_leaf(l)),
            # remove_empty_top_bracketing=True
            )
        t.set_label('ROOT')
        # collapse_unary(t)
        # chomsky_normal_form(t, factor="right")
        # chomsky_normal_form(t, factor="left")
        return t
        
# def rx_check(t):
#     if isinstance(t, str) == False:    
#         label = t.label()
#         print(label)
#         if isinstance(t[0], str) == False:
#             print(t[0].label)
#             if t[0].label == label:
#                 return True
#     return False

sh_rx = re.compile(r'^(.+?) -> \1')
def sh_rx_check(t):
    for p in t.productions():
        if sh_rx.search(str(p)):
            return True
    return False

def read_tree_rx(t):
    t = Tree.fromstring(
            t, 
            read_node=(lambda n: read_node(n)),  
            read_leaf=(lambda l: read_leaf(l)),
        )
    t.set_label('ROOT')
    if sh_rx_check(t) == True:
        return None
    return t

def left_corners(t):
    lcs = []
    for p in t.treepositions():
        if not any(p):
            lcs.append(t[p])
    del lcs[0]
    return lcs

def deep_rx_check(t):
    for root in t.subtrees(lambda t: t.height() > 2):
        for lc in left_corners(root):
            if not isinstance(lc, str):
                if lc.label() == root.label():
                    return True
    return False

def read_tree_deep_rx(t):
    t = Tree.fromstring(
            t, 
            read_node=(lambda n: read_node(n)),  
            read_leaf=(lambda l: read_leaf(l)),
        )
    t.set_label('ROOT')
    if deep_rx_check(t) == True:
        return None
    return t

def get_trees(line): # string -> [nltk.Tree] -- Tree.fromstring crashes with multiple trees per line, this does not.
    output = []
    try:
        trees = [read_tree_deep_rx(line)]
    except:
        line = '(' + line + ')'
        trees = read_tree_deep_rx(line)
    if trees:
        for tree in trees:
            if tree != None:
                output.append(tree)
    return output

### A Class for Actions ###

class Action:
    def __init__(self, act, rule):
        self.act = act
        self.rule = rule

    def __repr__(self):
        if self.act == 'ATTACH':
            return self.act
        else:
            return (self.act + ' ' + str(self.rule))

ATTACH = Action('ATTACH', None)

def execute(action, stack):
    if action.act == 'SHIFT':
        return [action.rule[0]] + stack
    if action.act == 'PROJECT':
        predicted_nts = [''.join(['m(' + nt + ')']) for nt in action.rule[1][1:]]
        return predicted_nts + [action.rule[0]] + stack[1:]
    if action.act == 'ATTACH':
        max_idx = len(stack) - 1
        i = 0
        while i < max_idx:
            if stack[i + 1] == ''.join(['m(', stack[i], ')']):
                return stack[:i] + stack[i+2:]
            else: i += 1
    if action.act == 'MATCH':
        shift = execute(Action('SHIFT', action.rule), stack)
        return execute(ATTACH, shift)
    if action.act == 'CONNECT':
        project = execute(Action('PROJECT', action.rule), stack)
        return execute(ATTACH, project)

def top_goal(stack):
    for s in stack:
        if 'm(' in s:
            return s

def guided_parse(actions):
    stack = ['m(S)']
    print(stack)
    for action in actions:
        stack = execute(action, stack)
        print(stack)

### A Class for Left-Corner Grammars ###

class LCG:
    def __init__(self, eager=False):
        self.eager = eager
        self.action_buffer = []
        self.d = defaultdict(lambda: defaultdict(int))

    def add_action(self, action):
        self.action_buffer = self.action_buffer + [action]

    def get_standard_actions(self, tree, top_down=True):
        if top_down == True:
            self.get_standard_actions(tree, top_down=False)
            self.add_action(ATTACH) # ATTACH
        if top_down == False:
            if tree.height() == 2:
                SHIFT = Action('SHIFT', (tree.label(), tree[0]))
                self.add_action(SHIFT) # SHIFT
            else:
                self.get_standard_actions(tree[0], top_down=False)
                PROJECT = Action('PROJECT', (tree.label(), [t.label() for t in tree[0:]]))
                self.add_action(PROJECT) #PROJECT
                for t in tree[1:]:
                    self.get_standard_actions(t, top_down=True)
        return self.action_buffer

    def get_eager_actions(self, tree, top_down=True):
        if top_down == True:
            if tree.height() == 2:
                 MATCH = Action('MATCH', ((tree.label(), tree[0])))
                 self.add_action(MATCH)
            else:
                self.get_eager_actions(tree[0], top_down=False)
                CONNECT = Action('CONNECT', (tree.label(), [t.label() for t in tree[0:]]))
                self.add_action(CONNECT)
                for t in tree[1:]:
                    self.get_eager_actions(t, top_down=True)
        if top_down == False:
            if tree.height() == 2:
                SHIFT = Action('SHIFT', ((tree.label(), tree[0])))
                self.add_action(SHIFT)
            else: 
                self.get_eager_actions(tree[0], top_down=False)
                PROJECT = Action('PROJECT', (tree.label(), [t.label() for t in tree[0:]]))
                self.add_action(PROJECT)
                for t in tree[1:]:
                    self.get_eager_actions(t, top_down=True)
        return self.action_buffer
    
    def get_actions(self, tree):
        if self.eager == False:
            output = self.get_standard_actions(tree)
        else:
            output = self.get_eager_actions(tree)
        return output

    def condition_action_pairs(self, tree):
        output = []
        self.action_buffer = [] # let us never forget the frustrating hours when I didn't realize self.action_buffer was not being cleared
        actions = self.get_actions(tree)
        primary_goal = 'm(' + tree.label() + ')'
        stack = [primary_goal]
        for action in actions:
            if stack:
                top_of_stack = stack[0]
                current_goal = top_goal(stack)
                condition = (top_of_stack, current_goal)
                output += [(condition, action)]
                stack = execute(action, stack)
        return output

    def train_d(self, corpus):
        with open(corpus) as lines:
            i = 0
            t = 0
            for line in lines:
                trees = get_trees(line)
                for tree in trees:
                    t +=1
                    pairs = self.condition_action_pairs(tree)
                    for pair in pairs:
                        self.d[str(pair[0])][str(pair[1])] += 1
                i += 1
                if i % 10000 == 0:
                    print("Training Progress: " + str(i) + " Lines Completed - " + str(t) + " Trees Processed")

def nested_dict_pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            nested_dict_pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def normalize(d): # count d -> prob d -- should work on any depth 2 nested dict that uses int as its leaves
    output = defaultdict(lambda: defaultdict(int))
    for cond, actions in d.items():
        tally = 0
        for action, count in actions.items():
            tally += count
        for action, count in actions.items():
            output[cond][action] = count/tally
    return output

def defaultify(d): # solution adapted from https://stackoverflow.com/a/50013806
    if not isinstance(d, dict):
        return d
    return defaultdict(lambda: None, {k: defaultify(v) for k, v in d.items()})

def dexport(d, name):
    with open('dfiles/' + name + '.json', 'w') as output:
        json.dump(d, output)

def dimport(dfile):
    if type(dfile) == str:
        with open('dfiles/' + dfile) as input:
            output = json.load(input)
            return defaultify(output)
    else:
        return dfile

### Evaluation ###

def tree_log_prob(tree, dfile, eager=False):
    d = dfile
    # d = dimport(dfile)
    # d = normalize(d)
    lcg = LCG(eager=eager)
    output = 0
    # if lexless == True:
    #     tree = tree._pformat_flat(nodesep="", parens="()", quotes=False)
    #     tree = Tree.fromstring(tree, 
    #         read_node=(lambda n: read_node(n)),  
    #         read_leaf=(lambda l: read_leaf_lexless(l)),
    #         remove_empty_top_bracketing=True
    #         )
    events = lcg.condition_action_pairs(tree)
    for event in events:
        cond = str(event[0])
        act = str(event[1])
        prob = d[cond][act]
        # with open('test_output.tsv', 'a') as out:                     # only uncomment if debugging
        #     out.write(cond + '\t' + act + '\t' + str(prob) +'\n')     # with small corpus
        if prob != 0: # in case the action is not in the dfile - can't take np.log(0)
            output += np.log(prob) 
    return output

def corpus_log_prob(corpus, dfile, eager=False):
    d = dimport(dfile)
    d = normalize(d)
    output = 0
    i = 0
    t = 0
    with open(corpus) as lines: 
        for line in lines:
            trees = get_trees(line)
            for tree in trees:
                tree_prob = tree_log_prob(tree, d, eager=eager)
                output += tree_prob
                t += 1
            i += 1
            if i % 10000 == 0:
                print("Log Prob Calculation Progress: " + str(i) + " Lines Completed - " + str(t) + " Trees Processed")
    return output

def paramcount(dfile):
    d = dimport(dfile)
    params = 0
    for cond, actions in d.items():
        params += len(actions)
        params -= 1
    return params

def aic(k, l):
    return (2 * k) - (2 * l)

def aic_complete(corpus, dfile, eager=False):
    k = paramcount(dfile)
    l = corpus_log_prob(corpus, dfile, eager)
    return (2 * k) - (2 * l)

def train_model(name, corpus, eager=False, calc_aic=True):
    print("Training in progress for " + name)
    assert isinstance(name, str), "Argument 'name' should be a string."
    assert isinstance(corpus, str), "Argument 'corpus' should be a string."
    model = LCG(eager)
    model.train_d(corpus)
    dexport(model.d, name)
    print('Model Training and Export Complete')
    if calc_aic == True:
        k = paramcount(model.d)
        l = corpus_log_prob(corpus, model.d, eager)
        a = aic(k, l)
        with open('dfiles/modelstats.csv', 'a') as output:
            output.write('\n' + name + ',' + str(k) + ',' + str(l) + ',' + str(a))
        print('AIC Calculation Complete')
        print('Model Parameters: ' + str(k))
        print('Model Log-Likelihood: ' + str(l))
        print('Model AIC: ' + str(a))

start_time = time.time()
train_model('AS-PLCG-DRX', 'full_ptb.txt', eager=False)
runtime = time.time() - start_time
print("Runtime: ", runtime)

start_time = time.time()
train_model('AE-PLCG-DRX', 'full_ptb.txt', eager=True)
runtime = time.time() - start_time
print("Runtime: ", runtime)

# For turning a bank of trees into latex friendly \Tree's
# def latex_print_list(file, n=10):
#     with open(file) as trees:
#         i = 0
#         while i < n:
#             for tree in trees:
#                 t = read_tree(tree)
#                 print(t.pformat_latex_qtree())
#                 i += 1

def lrc(tree):
    l,r,c = 0,0,0
    maxl, maxr, maxc = 0,0,0
    leafpos = tree.treepositions('leaves')
    for pos in leafpos:
        text = "".join([str(p) for p in pos])
        lb = re.findall(r"0+?1\b", text)
        if lb:
            l += len(lb)
            for item in lb:
                depth = item.count('0')
                if depth > maxl:
                    maxl = depth
        rb = re.findall(r"1+?0\b", text)
        if rb:
            r += len(rb)
            for item in rb:
                depth = item.count('1')
                if depth > maxr:
                    maxr = depth
        ce = re.findall(r"((?:10+?)+?1)\b", text) 
        if ce:
            c += len(ce)
            for item in ce:
                depth = item.count('10')
                if depth > maxc:
                    maxc = depth
    outlist = [l, r, c, maxl, maxr, maxc]
    output = "<comma>".join([str(num) for num in outlist])
    return output

def csv_safe(l):
    l = [str(i) for i in l]
    l = [i.replace(',',';') for i in l]
    l = [i.replace('<comma>',',') for i in l]
    return l

def action_splitter(action):
    i = 0
    if '(' in action:
        while action[i] != '(':
            i += 1
    return action[:i], action[i:]

def full_corpus_analysis(corpus, dfile, eager=False, lrca=False):
    modelname = dfile.replace('.json', '')
    corpusname = corpus.replace('.txt', '')
    compfile = 'analysis/' + modelname + '_' + corpusname + '_model_comparison.csv'
    actfile = 'analysis/' + modelname + '_' + corpusname + '_action_analysis.csv'
    print("Analysis Started")
    print("Corpus: " + corpusname)
    print("Model: " + modelname)
    print("Model Comparison File: " + compfile)
    print("Action Analysis File: " + actfile)
    lcg = LCG(eager=eager)
    d = dimport(dfile)
    d = normalize(d)
    with open(compfile, 'w') as comp_output:
        if lrca == True:
             comp_output.write('File, TreeNum, Tree, Sentence, L, R, C, MaxL, MaxR, MaxC, ' + modelname + '\n')
        else:
            comp_output.write('File, TreeNum, ' + modelname + '\n')
        with open(actfile, 'w') as act_output:
            act_output.write('Model, File, TreeNum, Tree, Condition, Action, Rule, Probability, Log-Prob, Cumulative Tree-Prob \n') # these files will be stacked
            totalll = 0
            i = 1
            t = 1
            with open(corpus) as corp:
                for line in corp:
                    trees = get_trees(line)
                    for tree in trees:
                        treetext = tree._pformat_flat(nodesep="", parens="()", quotes=False)
                        sentence = " ".join(tree.leaves()).replace(' .', '.')
                        leafpos = tree.treepositions('leaves')
                        treeprob = 0
                        events = lcg.condition_action_pairs(tree)
                        for event in events:
                            cond = str(event[0])
                            act = str(event[1])
                            prob = d[cond][act]
                            if prob == 0: 
                                logprob = 0 
                            else:
                                logprob = np.log(prob)
                            treeprob += logprob
                            actsplit = action_splitter(act)
                            act_outlist = [modelname, i, t, treetext, cond, actsplit[0], actsplit[1], prob, logprob, treeprob]
                            act_output.write(",".join(csv_safe(act_outlist)) + "\n")
                        if lrca == True: 
                            tlrc = lrc(tree)
                            comp_outlist = [i, t, treetext, sentence, tlrc, treeprob]
                        else:
                            comp_outlist = [i, t, treeprob]
                        comp_output.write(",".join(csv_safe(comp_outlist)) + '\n')
                        totalll += treeprob
                        t += 1
                    if i % 10000 == 0:
                        print("Files Analyzed: " + str(i))
                    i += 1
                if lrca == True:
                    comp_output.write('Total,,,,,,,,,,' + str(totalll))
                else:
                    comp_output.write('Total,,' + str(totalll))
                print(" ".join([modelname, corpusname, "Analysis Complete"]))

def print_action_lists(tree, standard=True, eager=True):
    lcg = LCG()
    if standard == True:
        print("Standard Actions:")
        lcg.get_standard_actions(tree)
        for action in lcg.action_buffer:
            print(action)
    lcg.action_buffer = []
    if standard == True:
        print("Eager Actions:")
        lcg.get_eager_actions(tree)
        for action in lcg.action_buffer:
            print(action)

def tex_event(e):
    e = str(e)
    e = e.replace("'m(","\m{").replace(")'","}").replace("'","")
    return e

def tex_action(a):
    act = str(a[0]).lower()
    rule = str(a[1])
    act = '\\textsc{' + act + '}'
    return act

def tree_analysis(t, std_dfile, eag_dfile):
    sd = dimport(std_dfile)
    ed = dimport(eag_dfile)
    slcg = LCG(eager=False)
    elcg = LCG(eager=True)
    sevents = slcg.condition_action_pairs(t)
    eevents = elcg.condition_action_pairs(t)
    for event in sevents:
        cond = str(event[0])
        act = str(event[1])
        prob = sd[cond][act]
        print("\t".join(cond,act,str(prob)))
    for event in sevents:
        cond = str(event[0])
        act = str(event[1])
        prob = sd[cond][act]
        print("\t".join(cond,act,str(prob)))