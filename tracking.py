import json

import numpy as np

class Tracker(object):
    """ Tracks dependencies between commands """

    def __init__(self, commands):
        """ commands is a list of pipebuild.Command objects """
        self.commands = commands

        self.update_ids()


    def update_ids(self):
        """
        Updates command_classes, which maps command classes to small integers
        """
        self.command_classes = {}
        classes = set([cmd.__class__ for cmd in self.commands])
        for (i, cls) in enumerate(classes):
            self.command_classes[cls] = i

    def compute_dependencies(self):
        """
        Computes a dependency graph for all commands created so far
        """
        N = len(self.commands)

        # Sparse array of filelists: list at (i, j) has all files that i
        # outputs that j depends on
        self.dependency_files = np.empty([N, N], dtype='object')

        for i in xrange(N):
            for j in xrange(N):
                self.dependency_files[i, j] = []

        for (i, early_command) in enumerate(self.commands):
            for (j, late_command) in enumerate(self.commands):
                # check if any of the early command's outputs are
                # inputs to the later command
                for outp in early_command.outfiles:
                    if outp in late_command.inputs:
                        self.dependency_files[i, j].append(outp)
                    else:
                        if outp.endswith('.txt') and 'arp' in late_command.comment:
                            pass

        # directed (acyclic) graph representing what tasks depend on what
        self.dependency_graph = np.vectorize(len)(self.dependency_files)

    def write_pipeline_to_json(self, filename):
        """
        Writes a graph of this pipeline (nodes, links) to a JSON file
        """
        # TODO compress nodes based on input-output relationships:
        # nodes with the same inputs and the same outputs can be compressed!
        self.compute_dependencies()
        stages = self.compute_stages()

        N = len(self.commands)
        nodes = []
        for (i, stage) in enumerate(stages):
            for (j, node) in enumerate(sorted(stage)):
                command = self.commands[node]
                n = {'name': command.comment,
                     'rawx': i,
                     'rawy': j,
                     'class': self.command_classes[command.__class__],
                     'id': 'node' + str(node)}
                nodes.append((node, n))

        nodes.sort()
        nodes = [n for (idnum, n) in nodes]
        links = []
        for i in xrange(N):
            for j in xrange(i+1, N):
                weight = int(self.dependency_graph[i, j])
                if weight != 0:
                    links.append({'source': i, 'target': j, 'value': weight})

        s = json.dumps({'nodes': nodes, 'links': links})
        with open(filename, 'w') as f:
            f.write(s)


    def compute_stages(self):
        """
        Breaks commands down into `stages' for visualization. Each stage only
        depends on events from previous stages.
        """
        all_stages = []
        this_stage_nodes = []
        next_stage_nodes = []

        # find all nodes without parents to start us off
        completed_nodes = set()
        for (i, cmd) in enumerate(self.commands):
            if np.sum(self.dependency_graph[:,i]) == 0:
                this_stage_nodes.append(i)

        # continue through each stage
        while True:
            next_stage_nodes = []
            for node in this_stage_nodes:
                completed_nodes.add(node)
                children = np.where(self.dependency_graph[node,:])[0]
                for child in children:
                    if child not in completed_nodes:
                        # make sure all its parents have been accounted for
                        # before adding it
                        parents = np.where(self.dependency_graph[:, child])[0]
                        if set(parents).issubset(completed_nodes):
                            next_stage_nodes.append(child)
            all_stages.append(this_stage_nodes)
            if next_stage_nodes == []:
                break
            this_stage_nodes = next_stage_nodes
        return all_stages

