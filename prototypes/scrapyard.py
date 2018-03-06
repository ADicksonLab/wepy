    # def get_closest_image(self, walker, images):

    #     # calculate distance from the walker to all given images
    #     dists = []
    #     for i, image in enumerate(images):
    #         dist = self.distance.distance(walker['positions'], image['positions'])
    #         dists.append(dist)

    #     # get the image that is the closest
    #     closest_image_idx = np.argmin(dists)

    #     return dists[closest_image_idx], closest_image_idx

    # def define_new_region(self, parent, walker):
    #     # the parents region assignment, which we will add to
    #     region_id = copy(parent.region_id)

    #     # the index of the new child/image/region
    #     child_idx = len(parent.children)

    #     # the new region identifier for the new region
    #     region_id.append(child_idx)

    #     # make a new node for the region, save the index of that
    #     # region as a child for the parent node in it's region_idx
    #     newnode = Node(region_id=region_id, positions=deepcopy(walker))
    #     parent.children.append(newnode)

    #     # return the index of the new region node as a child of the parent node
    #     return child_idx


    # # TODO old name: 'getdist'
    # def assign_walker(self, walker, level=0, region_assg=None):

    #     if region_assg is None:
    #         region_assg = []

    #     # children exist for the parent
    #     if len(parent.children) > 0:

    #         # find the closest image in this superregion (i.e. parent at current level)
    #         mindist, close_child_idx = self.get_closest_image(walker, parent.children)

    #         # if the minimum distance is over the cutoff we need to define new regions
    #         if mindist > self.max_region_sizes[level]:
    #             # if there are less children than the maximum number of images for this level
    #             if len(parent.children) < self.max_n_regions[level]:

    #                 # we can create a new region
    #                 close_child_idx = self.define_new_region(parent, walker)

    #         # the region assignment starts with this image/region index
    #         region_assg.append(close_child_idx)
    #         region_assg = self.assign_walkers(parent.children[close_child_idx], walker,
    #                                           level=level+1, region_assg=region_assg)

    #     # No children so we have to make them if we are not at the
    #     # bottom, define new regions for this node and make it a
    #     # parent
    #     elif level < len(self.max_region_sizes):
    #         # define a new region because none exist for this parent
    #         close_child_idx = self.define_new_region(parent, walker)

    #         # update the region_assg for the walker
    #         region_assg.append(close_child_idx)

    #         # recursively call getdist on the just created child one level deeper
    #         region_assg = self.assign_walkers(parent.children[close_child_idx], walker,
    #                                           level=level+1, region_assg=region_assg)

    #     return region_assg


# class Node(object):

#     def __init__(self, nwalkers=0, nreduc=0, nabovemin=0, children=[],
#                  region_id=None, to_clone=0, windx=[], positions=[]
#     ):
#         self.nwalkers = nwalkers
#         self.nreduc = nreduc
#         self.nabovemin = nabovemin
#         self.children = copy(children)
#         if region_id is None:
#             self.region_id = []
#         else:
#             self.region_id = region_id
#         self.toclone = to_clone
#         self.windx = copy(windx)
#         self.positions = copy(positions)

def branch_tree(self, parent_id, image):

        # add the new image to the image index
        image_idx = len(self.images)
        self.images.append(image)

        # get the parent for which we will make a new child for
        node = self.root_node
        for child_idx in parent_id:
            node = node.children[child_idx]

        # get the index of the new node starting the new chain
        new_child_idx = len(node.children)

        # once we have parent node we add a new branch down to the
        # bottom level
        for level in range(len(parent_id), self.n_levels):
            # make a new child node for the parent
            child_node = ImageNode(image_idx=image_idx, children=[])
            # add this as a child at the current level
            node.children.append(child_node)
            # move onto the next node
            node = child_node

        # return the new leaf node id
        leaf_node_id = list(parent_id)
        leaf_node_id = leaf_node_id + [0 for i in range(len(parent_id), self.n_levels)]
        leaf_node_id[len(parent_id)] += new_child_idx

        return tuple(leaf_node_id)


    def place_walker(self, walker):
        """Given the current hierarchy of Voronoi regions and the walker
        either assign it to a region or create a new region if
        necessary. This could mutate the state of the hierarchy.

        """

        # check to see if there are any images defined at all
        if self.root_node.image_idx is None:
            # if not we set the first images to the state of the first walker
            image_idx = len(self.images)
            # TODO distance image of the state is what should be
            # stored for the image

            # save the image of this walker for defining regions
            self.images.append(walker.state)

            # create the initial nodes for each level with this
            # initial image
            node = self.root_node
            node.image_idx = image_idx
            for level in range(self.n_levels):
                node = node.children[0]
                node.image_idx = image_idx

            assignment = tuple([0 for level in range(self.n_levels)])
            distances = tuple([0.0 for level in range(self.n_levels)])

        # if the image hierarchy has been initialized, assign this
        # walker to it
        else:
            # assign the walker to the defined regions
            assignment, distances = self.assign_walker(walker)

        # check each level's distance to see if it is beyond the
        # maximum allowed distance. If it is create a new branch
        # starting at the parent of the node at the level the walker
        # was closest to
        for level, distance in enumerate(distances):
            if distance > self.max_region_sizes[level]:
                image = walker.state
                parent_id = assignment[:level]
                assignment = self.branch_tree(parent_id, image)
                break

        return assignment


    def assign_walker(self, walker):

        assignment = []
        dists = []
        # perform a n-ary search through the hierarchy of regions by
        # performing a distance calculation to the images at each
        # level starting at the top
        node = self.root_node
        for level in range(self.n_levels):
            level_nodes = node.children

            # perform a distance calculation to all nodes at this
            # level
            image_dists = []
            for level_node in level_nodes:
                # get the image
                image = self.images[level_node.image_idx]
                dist = self.distance.distance(walker.state, image)
                image_dists.append(dist)

            # get the index of the image that is closest
            level_closest_child_idx = np.argmin(image_dists)
            # get the distance for the closest image
            level_closest_image_dist = image_dists[level_closest_child_idx]

            # save for return
            assignment.append(level_closest_child_idx)
            dists.append(level_closest_image_dist)

            # set this node as the next node
            node = level_nodes[level_closest_child_idx]

        return tuple(assignment), tuple(dists)

    def place_walkers(self, walkers):
        """Assign walkers to Voronoi regions. This function will add Voronoi
        cells to the hierarchy leaves if necessary, thus mutating the
        state of the resampler (i.e. the tree of Voronoi
        regions/images). This is the 'Scorer' in terms of the
        resampling framework.

        Returns a tuple of the region assignment for each level of the
        hierarchy for each walker.


        [(level_0_region_idx_0, level_1_region_idx_0, level_2_region_idx_0),
        (level_0_region_idx_1, level_1_region_idx_1, level_2_region_idx_1)]

        For two walkers with 3 levels, with the number of regions per
        level set at (2, 2, 2) (2 for each level) we might have:

        [(0, 1, 0),
         (0, 0, 0)
        ]

        """

        # region assignments
        walker_assignments = [None for i in range(len(walkers))]

        # assign walkers to regions
        for i, walker in enumerate(walkers):
            walker_assignments[i] = self.place_walker(walker)

        return walker_assignments


    def getmaxminwalk(self, children):
        #
        # This is a helper function for balancetree, which returns the
        # children with the highest and lowest numbers of walkers. As
        # well as the number of transitions that should occur to even them
        # out (ntrans).
        #

        minwalk = None
        maxwalk = None
        highchild = None
        lowchild = None

        for i, child in enumerate(children):
            # calculate the number of walkers that will be present
            # after cloning
            final_nwalkers = child.nwalkers + child.toclone
            # calculate the number of reducible walkers that will be
            # present after cloning
            final_nreduc = child.nreduc + child.toclone

            # update the maxwalk, which is the maximum number of
            # walkers that will exist after cloning, if the number of
            # reducible walkers is 1 or more
            if (not maxwalk or final_nwalkers > maxwalk) and (final_nreduc >= 1):
                maxwalk = final_nwalkers
                highchild = i

            # update the minwalk, which is the minimum number of
            # walkers that will exist after cloning, if the number of
            # number of walkers above the minimum weight is 1 or more
            if (not minwalk or final_nwalkers < minwalk) and (child.nabovemin >= 1):
                minwalk = final_nwalkers
                lowchild = i

        # if a minwalk and maxwalk are defined calculate ntrans, the
        # number of 'transitions' which is the number of walkers that
        # will be passed between them.
        if (minwalk and maxwalk):
            # the number of transitions is either the midpoint between
            # them (rounded down to an integer) or the sum of the
            # number of reducible walkers plus the number of toclones
            # for the walker with the highest number of walkers,
            # whichever is smaller
            ntrans = min(int((maxwalk - minwalk)/2),
                         children[highchild].nreduc + children[highchild].toclone)
        # initialize it to 0 if it is not set already
        else:
            ntrans = 0

        return minwalk, maxwalk, lowchild, highchild, ntrans


    def balancetree(self, parent):
        #
        # This is a recursive function, that balances each level of the image tree.
        # In each call, the argument is the parent node, and the balancing is done
        # between the children of that parent.
        #
        # The parent node passes on a "balancing debt" or surplus to its children
        # which is saved in the "toclone" variable
        #
        # The main purpose of this function is to update the following class variables:
        #     self.walkers_squashed
        #     self.num_clones
        # these are used in the resampler to write a resampling decision.
        #

        children = parent.children

        # this node has children, balance between them
        if len(children) > 1:

            # add the number of walkers to clone to both the number of
            # reducible walkers and the total number of walkers
            parent.nreduc += parent.toclone
            parent.nwalkers += parent.toclone

            # if there is a negative number of toclone the parent node
            # has a deficit; needs to merge walkers.
            if parent.toclone < 0:

                # find children that have reducible walkers
                for child in children:
                    final_nreduc = child.nreduc + child.toclone
                    if final_nreduc >= 1:
                        dif = min(abs(parent.toclone), final_nreduc)
                        parent.toclone += dif
                        child.toclone -= dif
                if parent.toclone < 0:
                    raise ValueError("Error! Children cannot pay their parent's debt")

            # if the number of toclone is greater than 0 the parent
            # has a surplus and needs to clone walkers
            if parent.toclone > 0:
                # find children that have walkers that can be cloned
                for child in children:
                    if child.nabovemin >= 1:
                        child.toclone += parent.toclone
                        parent.toclone = 0
                if parent.toclone > 0:
                    raise ValueError("Error! Children cannot clone walkers!")

            # balance between the children

            # find the nodes with the highest and lowest numbers of
            # walkers, and the number of transitions that are to occur
            # between them
            minwalk, maxwalk, lowchild, highchild, ntrans = self.getmaxminwalk(children)

            # if the number of transitions is 1 or more
            while (minwalk and maxwalk) and (ntrans >= 1):
                # merge walkers in highchild
                children[lowchild].toclone += ntrans
                # clone walkers in lowchild
                children[highchild].toclone -= ntrans
                # then recalculate the values
                minwalk, maxwalk, lowchild, highchild, ntrans = self.getmaxminwalk(children)

            # recursively call balancetree
            # children are as balanced as they are going to get
            # now run all children through balancetree
            for child in children:
                self.balancetree(child)

        # if there is only one child
        elif len(children) == 1:
            # only one child, just pass on the debt / surplus
            parent.nreduc += parent.toclone
            parent.nwalkers += parent.toclone
            children[0].toclone = parent.toclone
            parent.toclone = 0

            # recursively call balancetree
            self.balancetree(children[0])

        # no children, we are at the lowest level of the tree
        else:
            # figure out walkers to clone or merge

            # if the toclone debt is larger than the number of
            # reducible walkers we won't have enough walkers to merge
            if (-parent.toclone > parent.nreduc):
                raise ValueError("Error! node doesn't have enough walkers to merge")

            # merge and update until there is no debt
            while parent.toclone < 0:
                # MERGE: find the two walkers with the lowest weights

                # walker index of first walker
                r1 = None
                minwt = None
                for i in range(parent.nwalkers):
                    twt = self.walkerwt[parent.windx[i]]
                    if (r1 is None) or (twt < minwt):
                        minwt = twt
                        r1 = i

                # walker index of second walker
                r2 = None
                minwt = None
                for i in range(parent.nwalkers):
                    if i != r1:
                        twt = self.walkerwt[parent.windx[i]]
                        if (r2 is None) or (twt < minwt):
                            minwt = twt;
                            r2 = i


                r1index = parent.windx[r1]
                r2index = parent.windx[r2]
                # choose one of them to be the KEEPER and the other to
                # be squashed
                r3 = rand.random() * (self.walkerwt[r1index] + self.walkerwt[r2index])
                if r3 < self.walkerwt[r1index]:
                    keep_idx = r1index
                    squash_idx = r2index
                else:
                    keep_idx = r2index
                    squash_idx = r1index

                # move the weight
                self.walkerwt[keep_idx] += self.walkerwt[squash_idx]
                self.walkerwt[squash_idx] = 0

                # update the merge groups
                self.walkers_squashed[keep_idx] += [squash_idx] + self.walkers_squashed[squash_idx]
                self.walkers_squashed[squash_idx] = []

                # remove the squashed walker from the nodes that
                # reference it and account for it
                parent.windx.remove(squash_idx)
                parent.nwalkers -=1
                parent.toclone +=1

            # if there is no debt and rather an excess, clone until
            # there is no excess
            while parent.toclone > 0:
                # pick the one with the highest weight
                maxwt = None
                for i, w in enumerate(parent.windx):
                    twt = self.walkerwt[w]
                    if (maxwt is None) or twt > maxwt:
                        maxwt = twt;
                        r1walk = w

                self.num_clones[r1walk] += 1
                parent.nwalkers +=1
                parent.toclone -=1

    def n_walkers_per_region(assignments):
        # determine the number of walkers under each node

        n_region_walkers = {}
        # for each walker assignment record the count for each node it
        # is in in a dictionary of the node id tuples
        for assignment in assignments:
            node_id = ()
            # for each level of the assignment, count it for each
            # branch
            for level in assignment:
                node_id += (level,)

                n_region_walkers[node_id] += 1

        return n_region_walkers

    
    def populatetree(self, parent, level=0, region_assg=None, debug_prints=False):
        #
        # This function uses the walkerreg assignments to determine
        # how many walkers are under each node (nwalkers) at each level
        # of the tree. It also determines the number of those walkers
        # which are reducible (nreduc) as well as the number of walkers
        # that have weights above the minimum probability (nabovemin).
        #
        # If this is called at the "treetop" it will populate the entire
        # tree.
        #
        n_walkers = 0
        n_reducible_walkers = 0
        n_above_minp = 0

        if region_assg is None:
            parent_region_assg = []
        else:
            parent_region_assg = region_assg

        # if there are still levels in this hierarchy
        if level < len(self.max_region_sizes):

            # for the children under this parent, recursively call
            # this function. If they are a leaf node it will bottom
            # out and find the info on the walkers in that
            # leaf-node/region, else it will repeat until then
            for i, child in enumerate(parent.children):

                child_region_assg = copy(parent_region_assg)
                parent_region_assg.append(i)

                # recursively call
                tup = self.populate_tree(child, level=level+1,
                                         region_assg=parent_region_assg)
                child.n_walkers = tup[0]
                child.n_reducible_walkers = tup[1]
                child.n_above_minp = tup[2]

                n_walkers += tup[0]
                n_reducible_walkers += tup[1]
                n_above_minp += tup[2]

                if debug_prints:
                    print(child.region_id, *tup)

        # if we are at the bottom of the hierarchy, we want to save
        # which walkers are in these leaf nodes, and calculate the
        # number of reducible walkers, which can then be used to
        # balance the tree at a higher level
        else:

            leaf_walker_weights = []
            leaf_walker_idxs = []

            # save values for walkers in this leaf node/region/image
            for walker_idx in range(self.n_walkers):

                # if this walker is assigned to this region (we are at
                # the bottom here) then save information about the
                # walker in this node
                if self.walkerreg[walker_idx] == region_assg:

                    # increment the total count of walkers under this node (parent)
                    n_walkers += 1

                    # add this walker index to the list of walker
                    # indices under this node
                    leaf_walker_idxs.append(walker_idx)
                    # save the weight of this walker in the node
                    leaf_walker_weights.append(self.walkerwt[walker_idx])

                    # if this walker is above the minimum
                    # probability/weight increment the number of such
                    # walkers under the node
                    if self.walkerwt[walker_idx] > self.pmin:
                        n_above_minp += 1

            parent.windx = leaf_walker_idxs


            # calculate the number of reducible walkers under this
            # node, which is n_red = n_walkers - 1 with the constraint
            # that the largest possible merge is not greater than the
            # maximimum probability for a walker

            # must be more than 1 walker
            n_reducible_walkers = 0
            if n_walkers > 1:

                # figure out the most possible mergeable walkers
                # assuming they cannot ever be larger than pmax
                sorted_weights = list(np.sort(leaf_walker_weights))
                sum_weights = sorted_weights[0]
                for i in range(1, len(sorted_weights)):
                    sum_weights += sorted_weights[i]

                    # if we still haven't gone past pmax set the n_red
                    # to the current index
                    if sum_weights < self.pmax:
                        n_reducible_walkers = i

        return n_walkers, n_reducible_walkers, n_above_minp


    def settle_balance_old(self, leaf_parent):

        # merge groups and number of clones for just this leaf group
        merge_groups = [[] for i in self.walker_weights]
        walkers_num_clones = [0 for i in self.walker_weights]

        # within the last bunch of leaves we need to pay the leaf
        # parent's debt through cloning and merging
        leaves = self.children(leaf_parent)

        if self.node[leaf_parent]['balance'] < 0:
            # check to make sure that the debt has enough
            # mergeable walkers to merge to pay it
            assert (not -self.node[leaf_parent]['balance'] >
                    sum([self.node[leaf]['n_mergeable'] for leaf in leaves])), \
                            "Node doesn't have enough walkers to merge"

        # we will iterate through the children (either clongin or
        # merging) until the balance is settled
        leaf_it = iter(leaves)

        # if the balance is negative we merge
        while self.node[leaf_parent]['balance'] < 0:

            # get the leaf to do stuff with
            try:
                leaf = next(leaf_it)
            except StopIteration:
                # stop for this child and move onto the next
                break

            # find the two walkers with the lowest weight to merge
            weights = [self.walker_weights[i].weight for i in self.node[leaf]['walker_idxs']]

            # sort the weights and use to get the two lowest weight walkers

            walker_idxs = [i for i in self.node[leaf]['walker_idxs']
                           if i in np.argsort(weights)[:2]]

            # if the sum of these weights would be greater than
            # pmax move on to the next leaf to do merges
            if sum(np.array(weights)[walker_idxs]) > self.pmax:
                break

            # choose the one to keep the state of (e.g. KEEP_MERGE
            # in the Decision)
            keep_idx = rand.choice(walker_idxs)
            # get the other index for the squashed one
            squash_idx = walker_idxs[1 - walker_idxs.index(keep_idx)]

            # account for the weight from the squashed walker to
            # the keep walker
            self._walker_weights[keep_idx] += self.walker_weights[squash_idx]
            self._walker_weights[squash_idx] = 0.0

            # update the merge group
            merge_groups[keep_idx].append(squash_idx)

            # update the parent's balance
            self.node[leaf_parent]['balance'] += 1


        while self.node[leaf_parent]['balance'] > 0:

            # get the leaf to do stuff with
            try:
                leaf = next(leaf_it)
            except StopIteration:
                # stop for this child and move onto the next
                break

            weights = [self.walker_weights[i].weight for i in self.node[leaf]['walker_idxs']]

            # get the walker with the highest weight
            walker_idx = self.node[leaf]['walker_idxs'][np.argmax(weights)]

            # increase the number of clones assigned to this walker
            walkers_num_clones[walker_idx] += 1

            # update the parent's balance
            self.node[leaf_parent]['balance'] -= 1

        return merge_groups, walkers_num_clones
