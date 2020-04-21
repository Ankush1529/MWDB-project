from math import pow

def print_tree(root, height=0):
	if isinstance(root, dict):
		print_tree(root['left'], height+1)
		print_tree(root['right'], height+1)
        
        
def gini_index(set, labels):
	occur = float(sum([len(ele) for ele in set]))
	gini_score=score  = 0.0
	for ele in set:
		size = float(len(ele))
		if size == 0:
			continue
		for class_val in labels:
			p = [row[-1] for row in ele].count(class_val) / size
			score += pow(p,2)
		gini_score += (1.0 - score) * (size / occur)
	return gini_score


def test_split(index, value, data_set):
    left = list() 
    right =list()
    for row_val in data_set:
        if row_val[index] > value:
            right.append(row_val)
        else:
            left.append(row_val)
    return left, right

def get_split(data):
	c_values = list(set(row[-1] for row in data))
	b_index, b_value, b_score, b_groups = 3000, 3000, 3000, None
	for idx in range(len(data[0])-1):
		for row in data:
			groups = test_split(idx, row[idx], data)
			gini = gini_index(groups, c_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = idx, row[idx], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}


def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
        
        
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)



def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

        
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root