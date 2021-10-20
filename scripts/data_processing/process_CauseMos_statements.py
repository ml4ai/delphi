import os
import json
import pandas as pd

statements = []
evidences = []
adjective_frequencies = {'sub': {}, 'obj': {}}
_adjective_frequencies = {'sub': {}, 'obj': {}}
adjective_names = {'sub': {}, 'obj': {}}
_adjective_names = {'sub': {}, 'obj': {}}

adjective_pairs = {}
_adjective_pairs = {}

with open('../../data/causemos_indra_statements/CauseMos_indra_statements.json', 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines, 1):
        statement = json.loads(line)
        #print(json.dumps(statement, indent=4, sort_keys=True))
        belief = statement["_source"]["belief"]

        evidence = statement["_source"]["evidence"]

        for evid_idx, evid in enumerate(evidence, 1):
            text = evid["evidence_context"]["text"]

            _adjectives = []
            for key in ["subj_adjectives", "obj_adjectives"]:
                _adj = evid["evidence_context"][key]
                _adj = _adj if _adj else []
                _adjectives.append(_adj)

            _polarities = []
            for key in ["subj_polarity", "obj_polarity"]:
                _pol = evid["evidence_context"][key]
                _pol = _pol if _pol else 0
                _polarities.append(_pol)

            evidences.append({
                'Statement #': idx,
                'Evidence #': evid_idx,
                '_Sub Adj': ', '.join(_adjectives[0]),
                '_Obj Adj': ', '.join(_adjectives[1]),
                '_Sub Pol': _polarities[0],
                '_Obj Pol': _polarities[1],
                '# _Sub Adj': len(_adjectives[0]),
                '# _Obj Adj': len(_adjectives[1]),
                'Text': text
            })

            for idx2, key in enumerate(['sub', 'obj']):
                if len(_adjectives[idx2]) in _adjective_frequencies[key].keys():
                    _adjective_frequencies[key][len(_adjectives[idx2])] += 1
                else:
                    _adjective_frequencies[key][len(_adjectives[idx2])] = 1

            _adjectives[0] = ['None'] if len(_adjectives[0]) == 0 else _adjectives[0]
            _adjectives[1] = ['None'] if len(_adjectives[1]) == 0 else _adjectives[1]

            for adj in _adjectives[0]:
                if adj in _adjective_names['sub'].keys():
                    _adjective_names['sub'][adj] += 1
                else:
                    _adjective_names['sub'][adj] = 1

            for adj in _adjectives[1]:
                if adj in _adjective_names['obj'].keys():
                    _adjective_names['obj'][adj] += 1
                else:
                    _adjective_names['obj'][adj] = 1

            for sub in _adjectives[0]:
                for obj in _adjectives[1]:
                    adj_pair = (sub, obj)
                    if adj_pair in _adjective_pairs.keys():
                        _adjective_pairs[adj_pair] += 1
                    else:
                        _adjective_pairs[adj_pair] = 1

        #     print(len(evidence))
        #     print(json.dumps(statement, indent=4, sort_keys=True))
        #     exit()
        #
        # continue

        text = evidence[0]["evidence_context"]["text"]

        _adjectives = []
        for key in ["subj_adjectives", "obj_adjectives"]:
            _adj = evidence[0]["evidence_context"][key]
            _adj = _adj if _adj else []
            _adjectives.append(_adj)

        _polarities = []
        for key in ["subj_polarity", "obj_polarity"]:
            _pol = evidence[0]["evidence_context"][key]
            _pol = _pol if _pol else 0
            _polarities.append(_pol)

        concepts = []
        for key in ["subj", "obj"]:
            con = statement["_source"][key]["concept"]
            concepts.append(con)

        adjectives = []
        for key in ["subj", "obj"]:
            adj = statement["_source"][key]["adjectives"]
            adjectives.append(adj)

        polarities = []
        for key in ["subj", "obj"]:
            pol = statement["_source"][key]["polarity"]
            polarities.append(pol)

        statements.append({
            'Statement #': idx,
            'Belief': belief,
            'Subject': concepts[0],
            'Object': concepts[1],
            'Sub Adj': ', '.join(adjectives[0]),
            'Obj Adj': ', '.join(adjectives[1]),
            'Sub Pol': polarities[0],
            'Obj Pol': polarities[1],
            '_Sub Adj': ', '.join(_adjectives[0]),
            '_Obj Adj': ', '.join(_adjectives[1]),
            '_Sub Pol': _polarities[0],
            '_Obj Pol': _polarities[1],
            '# Sub Adj': len(adjectives[0]),
            '# Obj Adj': len(adjectives[1]),
            '# _Sub Adj': len(_adjectives[0]),
            '# _Obj Adj': len(_adjectives[1]),
            '# _Evidence': len(evidence),
            'Text': text
        })

        if len(adjectives[0]) > 1 or len(adjectives[1]) > 1:
            with open(f'../../data/causemos_indra_statements/multi_adjective/{idx}.json', 'w') as out:
                out.write(json.dumps(statement, indent=4, sort_keys=True))

        for idx2, key in enumerate(['sub', 'obj']):
            if len(adjectives[idx2]) in adjective_frequencies[key].keys():
                adjective_frequencies[key][len(adjectives[idx2])] += 1
            else:
                adjective_frequencies[key][len(adjectives[idx2])] = 1

        adjectives[0] = ['None'] if len(adjectives[0]) == 0 else adjectives[0]
        adjectives[1] = ['None'] if len(adjectives[1]) == 0 else adjectives[1]

        for adj in adjectives[0]:
            if adj in adjective_names['sub'].keys():
                adjective_names['sub'][adj] += 1
            else:
                adjective_names['sub'][adj] = 1

        for adj in adjectives[1]:
            if adj in adjective_names['obj'].keys():
                adjective_names['obj'][adj] += 1
            else:
                adjective_names['obj'][adj] = 1

        for sub in adjectives[0]:
            for obj in adjectives[1]:
                adj_pair = (sub, obj)
                if adj_pair in adjective_pairs.keys():
                    adjective_pairs[adj_pair] += 1
                else:
                    adjective_pairs[adj_pair] = 1


        # print(belief)
        # print(text)
        # print(_adjectives)
        # print(_polarities)
        # print(adjectives)
        # print(_polarities)
        # print(concepts)

df_statements = pd.DataFrame(statements)
df_evidences = pd.DataFrame(evidences)

df_statements.to_csv('../../data/causemos_indra_statements/statements.csv', index=False,
                     columns=['Statement #', 'Sub Adj', '_Sub Adj', 'Sub Pol', '_Sub Pol', 'Subject', 'Obj Adj',
                              '_Obj Adj', 'Obj Pol', '_Obj Pol', '# Sub Adj', '# _Sub Adj', '# Obj Adj', '# _Obj Adj',
                              '# _Evidence', 'Text'])
df_evidences.to_csv('../../data/causemos_indra_statements/evidence.csv', index=False,
                    columns=['Statement #', 'Evidence #', '_Sub Adj', '_Sub Pol', '_Obj Adj', '_Obj Pol', '# _Sub Adj',
                             '# _Obj Adj', 'Text'])

# df_sub_adj_counts = df_statements.groupby(by='# Sub Adj').count()
# df_obj_adj_counts = df_statements.groupby(by='# Obj Adj').count()
#
# _df_sub_adj_counts = df_statements.groupby(by='# _Sub Adj').count()
# _df_obj_adj_counts = df_statements.groupby(by='# _Obj Adj').count()
#
# df_sub_adj_counts.to_csv('../../data/causemos_indra_statements/sub_adj_counts.csv', index=False)
# df_obj_adj_counts.to_csv('../../data/causemos_indra_statements/obj_adj_counts.csv', index=False)
#
# _df_sub_adj_counts.to_csv('../../data/causemos_indra_statements/_sub_adj_counts.csv', index=False)
# _df_obj_adj_counts.to_csv('../../data/causemos_indra_statements/_obj_adj_counts.csv', index=False)


for idx2, key in enumerate(['sub', 'obj']):
    multiplicity = []
    frequency = []
    for mult, freq in adjective_frequencies[key].items():
        multiplicity.append(mult)
        frequency.append(freq)

    df_freq = pd.DataFrame({'# Adjectives': multiplicity, 'frequency': frequency})
    df_freq.to_csv(f'../../data/causemos_indra_statements/{key}_adj_counts.csv', index=False)

    multiplicity = []
    frequency = []
    for mult, freq in _adjective_frequencies[key].items():
        multiplicity.append(mult)
        frequency.append(freq)

    df_freq = pd.DataFrame({'# Adjectives': multiplicity, 'frequency': frequency})
    df_freq.to_csv(f'../../data/causemos_indra_statements/_{key}_adj_counts.csv', index=False)

    adjective = []
    frequency = []
    for adj, freq in adjective_names[key].items():
        adjective.append(adj)
        frequency.append(freq)

    df_freq = pd.DataFrame({'Adjective': adjective, 'frequency': frequency})
    df_freq.to_csv(f'../../data/causemos_indra_statements/{key}_adjectives.csv', index=False)

    adjective = []
    frequency = []
    for adj, freq in _adjective_names[key].items():
        adjective.append(adj)
        frequency.append(freq)

    df_freq = pd.DataFrame({'Adjective': adjective, 'frequency': frequency})
    df_freq.to_csv(f'../../data/causemos_indra_statements/_{key}_adjectives.csv', index=False)


sub = []
obj = []
freq = []
for adj_pair, count in adjective_pairs.items():
    sub.append(adj_pair[0])
    obj.append(adj_pair[1])
    freq.append(count)

df_pair = pd.DataFrame({'Subject': sub, 'Object': obj, 'frequency': freq})
df_pair.to_csv(f'../../data/causemos_indra_statements/adjective_pairs.csv', index=False)


sub = []
obj = []
freq = []
for adj_pair, count in _adjective_pairs.items():
    sub.append(adj_pair[0])
    obj.append(adj_pair[1])
    freq.append(count)

df_pair = pd.DataFrame({'Subject': sub, 'Object': obj, 'frequency': freq})
df_pair.to_csv(f'../../data/causemos_indra_statements/_adjective_pairs.csv', index=False)

